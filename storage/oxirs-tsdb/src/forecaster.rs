//! Time-series forecasting using exponential smoothing and moving averages.
//!
//! Implements three forecasting methods:
//! - **Simple Exponential Smoothing (SES)**: `s_t = α·y_t + (1−α)·s_{t-1}`
//! - **Holt's Double Exponential Smoothing**: level + trend components
//! - **Simple Moving Average**: mean of the last `window` observations
//!
//! Reference: Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed.

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by forecasting operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ForecastError {
    /// The input data is empty or too short for the chosen method.
    InsufficientData { required: usize, provided: usize },
    /// A smoothing parameter is outside the valid range (0, 1).
    InvalidAlpha(f64),
    /// A smoothing parameter β is outside the valid range (0, 1).
    InvalidBeta(f64),
    /// Window size must be ≥ 1.
    InvalidWindow(usize),
    /// `fit` must be called before `predict` or `residuals`.
    NotFitted,
}

impl std::fmt::Display for ForecastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForecastError::InsufficientData { required, provided } => write!(
                f,
                "Insufficient data: required {}, provided {}",
                required, provided
            ),
            ForecastError::InvalidAlpha(a) => {
                write!(f, "Alpha {} is outside (0, 1)", a)
            }
            ForecastError::InvalidBeta(b) => write!(f, "Beta {} is outside (0, 1)", b),
            ForecastError::InvalidWindow(w) => write!(f, "Window {} must be >= 1", w),
            ForecastError::NotFitted => write!(f, "Model has not been fitted yet"),
        }
    }
}

impl std::error::Error for ForecastError {}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single time-series observation.
#[derive(Debug, Clone, PartialEq)]
pub struct DataPoint {
    /// Unix-epoch millisecond timestamp.
    pub timestamp: u64,
    /// Observed value.
    pub value: f64,
}

impl DataPoint {
    /// Create a new data point.
    pub fn new(timestamp: u64, value: f64) -> Self {
        Self { timestamp, value }
    }
}

/// Which smoothing algorithm to use.
#[derive(Debug, Clone, PartialEq)]
pub enum SmoothingMethod {
    /// Simple Exponential Smoothing — level only.
    Simple {
        /// Level smoothing factor ∈ (0, 1).
        alpha: f64,
    },
    /// Holt's Double Exponential Smoothing — level + linear trend.
    Holt {
        /// Level smoothing factor ∈ (0, 1).
        alpha: f64,
        /// Trend smoothing factor ∈ (0, 1).
        beta: f64,
    },
    /// Simple Moving Average over the last `window` points.
    MovingAverage {
        /// Number of observations to average.
        window: usize,
    },
}

/// The result of a forecasting run.
#[derive(Debug, Clone, PartialEq)]
pub struct ForecastResult {
    /// Number of steps ahead predicted.
    pub horizon: usize,
    /// Predicted `DataPoint`s (one per horizon step).
    pub predictions: Vec<DataPoint>,
    /// Optional 95% confidence interval `(lower, upper)` applied uniformly to
    /// all predictions (simplified).
    pub confidence_interval: Option<(f64, f64)>,
    /// Human-readable name of the method used.
    pub method: String,
}

// ---------------------------------------------------------------------------
// Internal fitted state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum FittedState {
    Simple {
        alpha: f64,
        /// Smoothed level values aligned with training data.
        smoothed: Vec<f64>,
        /// Last smoothed level (used for prediction).
        last_level: f64,
    },
    Holt {
        alpha: f64,
        beta: f64,
        smoothed: Vec<f64>,
        last_level: f64,
        last_trend: f64,
    },
    MovingAverage {
        window: usize,
        smoothed: Vec<f64>,
        /// The last `window` training values for carrying forward.
        recent_values: Vec<f64>,
    },
}

// ---------------------------------------------------------------------------
// Forecaster
// ---------------------------------------------------------------------------

/// A univariate time-series forecaster.
pub struct Forecaster {
    method: SmoothingMethod,
    state: Option<FittedState>,
}

impl Forecaster {
    /// Create a new forecaster with the chosen smoothing method.
    pub fn new(method: SmoothingMethod) -> Self {
        Self {
            method,
            state: None,
        }
    }

    // -----------------------------------------------------------------------
    // fit
    // -----------------------------------------------------------------------

    /// Fit the model on historical data.
    ///
    /// The data must be sorted by timestamp (earliest first).  At least one
    /// observation is required; Holt requires at least two.
    pub fn fit(&mut self, data: &[DataPoint]) -> Result<(), ForecastError> {
        match &self.method {
            SmoothingMethod::Simple { alpha } => self.fit_simple(*alpha, data),
            SmoothingMethod::Holt { alpha, beta } => self.fit_holt(*alpha, *beta, data),
            SmoothingMethod::MovingAverage { window } => self.fit_ma(*window, data),
        }
    }

    fn fit_simple(&mut self, alpha: f64, data: &[DataPoint]) -> Result<(), ForecastError> {
        validate_alpha(alpha)?;
        if data.is_empty() {
            return Err(ForecastError::InsufficientData {
                required: 1,
                provided: 0,
            });
        }

        let mut smoothed = Vec::with_capacity(data.len());
        // Initialise with the first observation.
        let mut s = data[0].value;
        smoothed.push(s);

        for dp in data.iter().skip(1) {
            s = alpha * dp.value + (1.0 - alpha) * s;
            smoothed.push(s);
        }

        self.state = Some(FittedState::Simple {
            alpha,
            last_level: s,
            smoothed,
        });
        Ok(())
    }

    fn fit_holt(&mut self, alpha: f64, beta: f64, data: &[DataPoint]) -> Result<(), ForecastError> {
        validate_alpha(alpha)?;
        validate_beta(beta)?;
        if data.len() < 2 {
            return Err(ForecastError::InsufficientData {
                required: 2,
                provided: data.len(),
            });
        }

        let mut smoothed = Vec::with_capacity(data.len());
        // Initialise level with first value, trend with the first difference.
        let mut level = data[0].value;
        let mut trend = data[1].value - data[0].value;
        smoothed.push(level + trend);

        for dp in data.iter().skip(1) {
            let prev_level = level;
            level = alpha * dp.value + (1.0 - alpha) * (level + trend);
            trend = beta * (level - prev_level) + (1.0 - beta) * trend;
            smoothed.push(level + trend);
        }

        self.state = Some(FittedState::Holt {
            alpha,
            beta,
            smoothed,
            last_level: level,
            last_trend: trend,
        });
        Ok(())
    }

    fn fit_ma(&mut self, window: usize, data: &[DataPoint]) -> Result<(), ForecastError> {
        if window == 0 {
            return Err(ForecastError::InvalidWindow(window));
        }
        if data.is_empty() {
            return Err(ForecastError::InsufficientData {
                required: 1,
                provided: 0,
            });
        }

        let values: Vec<f64> = data.iter().map(|dp| dp.value).collect();
        let mut smoothed = Vec::with_capacity(data.len());

        for i in 0..data.len() {
            let start = (i + 1).saturating_sub(window);
            let slice = &values[start..=i];
            let mean = slice.iter().sum::<f64>() / slice.len() as f64;
            smoothed.push(mean);
        }

        let recent_start = if data.len() >= window {
            data.len() - window
        } else {
            0
        };
        let recent_values = values[recent_start..].to_vec();

        self.state = Some(FittedState::MovingAverage {
            window,
            smoothed,
            recent_values,
        });
        Ok(())
    }

    // -----------------------------------------------------------------------
    // predict
    // -----------------------------------------------------------------------

    /// Forecast `horizon` steps into the future.
    ///
    /// Each step is `interval_ms` milliseconds after `last_timestamp`.
    pub fn predict(
        &self,
        horizon: usize,
        last_timestamp: u64,
        interval_ms: u64,
    ) -> Result<ForecastResult, ForecastError> {
        let state = self.state.as_ref().ok_or(ForecastError::NotFitted)?;

        let (predictions, method_name) = match state {
            FittedState::Simple { last_level, .. } => {
                // SES: flat forecast (level stays constant)
                let preds = (0..horizon)
                    .map(|h| DataPoint {
                        timestamp: last_timestamp + (h as u64 + 1) * interval_ms,
                        value: *last_level,
                    })
                    .collect();
                (preds, "simple_exponential_smoothing".to_string())
            }
            FittedState::Holt {
                last_level,
                last_trend,
                ..
            } => {
                // Holt: linear projection
                let preds = (0..horizon)
                    .map(|h| {
                        let h_f = h as f64 + 1.0;
                        DataPoint {
                            timestamp: last_timestamp + (h as u64 + 1) * interval_ms,
                            value: last_level + h_f * last_trend,
                        }
                    })
                    .collect();
                (preds, "holt_double_exponential_smoothing".to_string())
            }
            FittedState::MovingAverage {
                window,
                recent_values,
                ..
            } => {
                // MA: rolling mean forecast
                let mut buf = recent_values.clone();
                let mut preds = Vec::with_capacity(horizon);
                for h in 0..horizon {
                    let w = (*window).min(buf.len());
                    let start = buf.len() - w;
                    let mean = buf[start..].iter().sum::<f64>() / w as f64;
                    preds.push(DataPoint {
                        timestamp: last_timestamp + (h as u64 + 1) * interval_ms,
                        value: mean,
                    });
                    buf.push(mean);
                }
                (preds, format!("moving_average({})", window))
            }
        };

        Ok(ForecastResult {
            horizon,
            predictions,
            confidence_interval: None,
            method: method_name,
        })
    }

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------

    /// Compute in-sample residuals (actual − predicted) for each training point.
    ///
    /// Returns an empty vec if the model has not been fitted.
    pub fn residuals(&self, data: &[DataPoint]) -> Vec<f64> {
        let state = match &self.state {
            Some(s) => s,
            None => return vec![],
        };
        let smoothed = match state {
            FittedState::Simple { smoothed, .. } => smoothed,
            FittedState::Holt { smoothed, .. } => smoothed,
            FittedState::MovingAverage { smoothed, .. } => smoothed,
        };
        data.iter()
            .zip(smoothed.iter())
            .map(|(dp, s)| dp.value - s)
            .collect()
    }

    /// Return the smoothing parameters used during fitting, if the model has
    /// been fitted.
    ///
    /// - `Simple`: returns `(alpha, None)`.
    /// - `Holt`: returns `(alpha, Some(beta))`.
    /// - `MovingAverage`: returns `(0.0, None)` (no alpha/beta parameters).
    pub fn smoothing_params(&self) -> Option<(f64, Option<f64>)> {
        match &self.state {
            None => None,
            Some(FittedState::Simple { alpha, .. }) => Some((*alpha, None)),
            Some(FittedState::Holt { alpha, beta, .. }) => Some((*alpha, Some(*beta))),
            Some(FittedState::MovingAverage { .. }) => Some((0.0, None)),
        }
    }

    /// Mean Absolute Error of the in-sample fitted values.
    pub fn mae(&self, data: &[DataPoint]) -> f64 {
        let residuals = self.residuals(data);
        if residuals.is_empty() {
            return 0.0;
        }
        residuals.iter().map(|r| r.abs()).sum::<f64>() / residuals.len() as f64
    }

    /// Root Mean Squared Error of the in-sample fitted values.
    pub fn rmse(&self, data: &[DataPoint]) -> f64 {
        let residuals = self.residuals(data);
        if residuals.is_empty() {
            return 0.0;
        }
        let mse = residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64;
        mse.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_alpha(alpha: f64) -> Result<(), ForecastError> {
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(ForecastError::InvalidAlpha(alpha));
    }
    Ok(())
}

fn validate_beta(beta: f64) -> Result<(), ForecastError> {
    if beta <= 0.0 || beta >= 1.0 {
        return Err(ForecastError::InvalidBeta(beta));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear(n: usize, slope: f64, intercept: f64) -> Vec<DataPoint> {
        (0..n)
            .map(|i| DataPoint::new(i as u64 * 1000, intercept + slope * i as f64))
            .collect()
    }

    fn make_constant(n: usize, val: f64) -> Vec<DataPoint> {
        (0..n)
            .map(|i| DataPoint::new(i as u64 * 1000, val))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Simple Exponential Smoothing
    // -----------------------------------------------------------------------

    #[test]
    fn test_simple_fit_single_point() {
        let mut f = Forecaster::new(SmoothingMethod::Simple { alpha: 0.3 });
        let data = vec![DataPoint::new(0, 10.0)];
        assert!(f.fit(&data).is_ok());
    }

    #[test]
    fn test_simple_fit_constant_series() {
        let data = make_constant(10, 5.0);
        let mut f = Forecaster::new(SmoothingMethod::Simple { alpha: 0.5 });
        f.fit(&data).unwrap();
        let pred = f.predict(1, 9000, 1000).unwrap();
        // For a constant series the forecast should be ≈ 5.0
        let diff = (pred.predictions[0].value - 5.0).abs();
        assert!(
            diff < 1e-9,
            "Expected ~5.0, got {}",
            pred.predictions[0].value
        );
    }

    #[test]
    fn test_simple_predict_horizon() {
        let data = make_constant(5, 3.0);
        let mut f = Forecaster::new(SmoothingMethod::Simple { alpha: 0.5 });
        f.fit(&data).unwrap();
        let result = f.predict(4, 4000, 1000).unwrap();
        assert_eq!(result.predictions.len(), 4);
        assert_eq!(result.horizon, 4);
    }

    #[test]
    fn test_simple_predict_timestamps() {
        let data = make_constant(3, 1.0);
        let mut f = Forecaster::new(SmoothingMethod::Simple { alpha: 0.3 });
        f.fit(&data).unwrap();
        let result = f.predict(3, 10_000, 500).unwrap();
        assert_eq!(result.predictions[0].timestamp, 10_500);
        assert_eq!(result.predictions[1].timestamp, 11_000);
        assert_eq!(result.predictions[2].timestamp, 11_500);
    }

    #[test]
    fn test_simple_method_name() {
        let data = make_constant(2, 1.0);
        let mut f = Forecaster::new(SmoothingMethod::Simple { alpha: 0.5 });
        f.fit(&data).unwrap();
        let result = f.predict(1, 1000, 1000).unwrap();
        assert!(result.method.contains("simple"));
    }

    #[test]
    fn test_simple_invalid_alpha_zero() {
        let mut f = Forecaster::new(SmoothingMethod::Simple { alpha: 0.0 });
        let err = f.fit(&make_constant(3, 1.0));
        assert!(matches!(err, Err(ForecastError::InvalidAlpha(_))));
    }

    #[test]
    fn test_simple_invalid_alpha_one() {
        let mut f = Forecaster::new(SmoothingMethod::Simple { alpha: 1.0 });
        let err = f.fit(&make_constant(3, 1.0));
        assert!(matches!(err, Err(ForecastError::InvalidAlpha(_))));
    }

    #[test]
    fn test_simple_empty_data_error() {
        let mut f = Forecaster::new(SmoothingMethod::Simple { alpha: 0.3 });
        let err = f.fit(&[]);
        assert!(matches!(err, Err(ForecastError::InsufficientData { .. })));
    }

    // -----------------------------------------------------------------------
    // Holt Double Exponential Smoothing
    // -----------------------------------------------------------------------

    #[test]
    fn test_holt_fit_linear_series() {
        let data = make_linear(10, 2.0, 0.0); // 0, 2, 4, 6, …
        let mut f = Forecaster::new(SmoothingMethod::Holt {
            alpha: 0.5,
            beta: 0.3,
        });
        assert!(f.fit(&data).is_ok());
    }

    #[test]
    fn test_holt_predict_horizon() {
        let data = make_linear(5, 1.0, 0.0);
        let mut f = Forecaster::new(SmoothingMethod::Holt {
            alpha: 0.5,
            beta: 0.3,
        });
        f.fit(&data).unwrap();
        let result = f.predict(3, 4000, 1000).unwrap();
        assert_eq!(result.predictions.len(), 3);
    }

    #[test]
    fn test_holt_predicts_upward_trend() {
        let data = make_linear(10, 1.0, 0.0);
        let mut f = Forecaster::new(SmoothingMethod::Holt {
            alpha: 0.8,
            beta: 0.5,
        });
        f.fit(&data).unwrap();
        let result = f.predict(3, 9000, 1000).unwrap();
        // For a strongly increasing series each successive prediction should be ≥ prev.
        let preds = &result.predictions;
        assert!(preds[1].value >= preds[0].value - 0.1);
        assert!(preds[2].value >= preds[1].value - 0.1);
    }

    #[test]
    fn test_holt_insufficient_data_error() {
        let mut f = Forecaster::new(SmoothingMethod::Holt {
            alpha: 0.5,
            beta: 0.3,
        });
        let err = f.fit(&[DataPoint::new(0, 1.0)]);
        assert!(matches!(
            err,
            Err(ForecastError::InsufficientData { required: 2, .. })
        ));
    }

    #[test]
    fn test_holt_invalid_beta_error() {
        let mut f = Forecaster::new(SmoothingMethod::Holt {
            alpha: 0.5,
            beta: 0.0,
        });
        let err = f.fit(&make_linear(3, 1.0, 0.0));
        assert!(matches!(err, Err(ForecastError::InvalidBeta(_))));
    }

    #[test]
    fn test_holt_method_name() {
        let data = make_linear(3, 1.0, 0.0);
        let mut f = Forecaster::new(SmoothingMethod::Holt {
            alpha: 0.5,
            beta: 0.3,
        });
        f.fit(&data).unwrap();
        let result = f.predict(1, 2000, 1000).unwrap();
        assert!(result.method.contains("holt"));
    }

    // -----------------------------------------------------------------------
    // Moving Average
    // -----------------------------------------------------------------------

    #[test]
    fn test_ma_fit_basic() {
        let data = make_constant(5, 4.0);
        let mut f = Forecaster::new(SmoothingMethod::MovingAverage { window: 3 });
        assert!(f.fit(&data).is_ok());
    }

    #[test]
    fn test_ma_predict_constant() {
        let data = make_constant(5, 7.0);
        let mut f = Forecaster::new(SmoothingMethod::MovingAverage { window: 3 });
        f.fit(&data).unwrap();
        let result = f.predict(2, 4000, 1000).unwrap();
        for p in &result.predictions {
            let diff = (p.value - 7.0).abs();
            assert!(diff < 1e-9, "Expected 7.0, got {}", p.value);
        }
    }

    #[test]
    fn test_ma_predict_horizon_length() {
        let data = make_linear(10, 1.0, 0.0);
        let mut f = Forecaster::new(SmoothingMethod::MovingAverage { window: 3 });
        f.fit(&data).unwrap();
        let result = f.predict(5, 9000, 1000).unwrap();
        assert_eq!(result.predictions.len(), 5);
    }

    #[test]
    fn test_ma_window_larger_than_data() {
        let data = make_constant(3, 2.0);
        let mut f = Forecaster::new(SmoothingMethod::MovingAverage { window: 10 });
        f.fit(&data).unwrap();
        let result = f.predict(1, 2000, 1000).unwrap();
        let diff = (result.predictions[0].value - 2.0).abs();
        assert!(diff < 1e-9);
    }

    #[test]
    fn test_ma_method_name() {
        let data = make_constant(3, 1.0);
        let mut f = Forecaster::new(SmoothingMethod::MovingAverage { window: 2 });
        f.fit(&data).unwrap();
        let result = f.predict(1, 2000, 1000).unwrap();
        assert!(result.method.contains("moving_average"));
    }

    #[test]
    fn test_ma_zero_window_error() {
        let mut f = Forecaster::new(SmoothingMethod::MovingAverage { window: 0 });
        let err = f.fit(&make_constant(3, 1.0));
        assert!(matches!(err, Err(ForecastError::InvalidWindow(0))));
    }

    #[test]
    fn test_ma_empty_data_error() {
        let mut f = Forecaster::new(SmoothingMethod::MovingAverage { window: 3 });
        let err = f.fit(&[]);
        assert!(matches!(err, Err(ForecastError::InsufficientData { .. })));
    }

    // -----------------------------------------------------------------------
    // Residuals
    // -----------------------------------------------------------------------

    #[test]
    fn test_residuals_length() {
        let data = make_constant(5, 3.0);
        let mut f = Forecaster::new(SmoothingMethod::Simple { alpha: 0.5 });
        f.fit(&data).unwrap();
        let res = f.residuals(&data);
        assert_eq!(res.len(), 5);
    }

    #[test]
    fn test_residuals_constant_series_first_zero() {
        let data = make_constant(5, 10.0);
        let mut f = Forecaster::new(SmoothingMethod::Simple { alpha: 0.5 });
        f.fit(&data).unwrap();
        let res = f.residuals(&data);
        assert!((res[0]).abs() < 1e-9);
    }

    #[test]
    fn test_residuals_unfitted_returns_empty() {
        let f = Forecaster::new(SmoothingMethod::Simple { alpha: 0.3 });
        let data = make_constant(3, 1.0);
        let res = f.residuals(&data);
        assert!(res.is_empty());
    }

    // -----------------------------------------------------------------------
    // MAE / RMSE
    // -----------------------------------------------------------------------

    #[test]
    fn test_mae_constant_is_zero() {
        let data = make_constant(5, 5.0);
        let mut f = Forecaster::new(SmoothingMethod::Simple { alpha: 0.5 });
        f.fit(&data).unwrap();
        let mae = f.mae(&data);
        assert!(
            mae < 1e-9,
            "MAE should be 0 for constant series, got {}",
            mae
        );
    }

    #[test]
    fn test_rmse_constant_is_zero() {
        let data = make_constant(5, 5.0);
        let mut f = Forecaster::new(SmoothingMethod::Simple { alpha: 0.5 });
        f.fit(&data).unwrap();
        let rmse = f.rmse(&data);
        assert!(rmse < 1e-9);
    }

    #[test]
    fn test_mae_positive() {
        let data = make_linear(10, 2.0, 0.0);
        let mut f = Forecaster::new(SmoothingMethod::Simple { alpha: 0.3 });
        f.fit(&data).unwrap();
        let mae = f.mae(&data);
        assert!(mae >= 0.0);
    }

    #[test]
    fn test_rmse_positive() {
        let data = make_linear(10, 2.0, 0.0);
        let mut f = Forecaster::new(SmoothingMethod::Simple { alpha: 0.3 });
        f.fit(&data).unwrap();
        let rmse = f.rmse(&data);
        assert!(rmse >= 0.0);
    }

    #[test]
    fn test_rmse_ge_mae() {
        // RMSE ≥ MAE always
        let data = make_linear(10, 3.0, 1.0);
        let mut f = Forecaster::new(SmoothingMethod::Simple { alpha: 0.4 });
        f.fit(&data).unwrap();
        let mae = f.mae(&data);
        let rmse = f.rmse(&data);
        assert!(rmse >= mae - 1e-9);
    }

    #[test]
    fn test_mae_unfitted_returns_zero() {
        let f = Forecaster::new(SmoothingMethod::Simple { alpha: 0.3 });
        let data = make_constant(3, 1.0);
        assert_eq!(f.mae(&data), 0.0);
    }

    #[test]
    fn test_rmse_unfitted_returns_zero() {
        let f = Forecaster::new(SmoothingMethod::Simple { alpha: 0.3 });
        let data = make_constant(3, 1.0);
        assert_eq!(f.rmse(&data), 0.0);
    }

    // -----------------------------------------------------------------------
    // predict before fit
    // -----------------------------------------------------------------------

    #[test]
    fn test_predict_before_fit_error() {
        let f = Forecaster::new(SmoothingMethod::Simple { alpha: 0.3 });
        let err = f.predict(1, 0, 1000);
        assert_eq!(err, Err(ForecastError::NotFitted));
    }

    // -----------------------------------------------------------------------
    // ForecastError display
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_display() {
        let e = ForecastError::NotFitted;
        assert!(!e.to_string().is_empty());
        let e2 = ForecastError::InvalidAlpha(1.5);
        assert!(e2.to_string().contains("1.5"));
    }

    // -----------------------------------------------------------------------
    // smoothing_params
    // -----------------------------------------------------------------------

    #[test]
    fn test_smoothing_params_unfitted_returns_none() {
        let f = Forecaster::new(SmoothingMethod::Simple { alpha: 0.4 });
        assert_eq!(f.smoothing_params(), None);
    }

    #[test]
    fn test_smoothing_params_simple_returns_alpha() {
        let data = make_constant(3, 1.0);
        let mut f = Forecaster::new(SmoothingMethod::Simple { alpha: 0.4 });
        f.fit(&data).unwrap();
        let params = f.smoothing_params().unwrap();
        assert!((params.0 - 0.4).abs() < 1e-12);
        assert!(params.1.is_none());
    }

    #[test]
    fn test_smoothing_params_holt_returns_alpha_beta() {
        let data = make_linear(4, 1.0, 0.0);
        let mut f = Forecaster::new(SmoothingMethod::Holt {
            alpha: 0.6,
            beta: 0.2,
        });
        f.fit(&data).unwrap();
        let params = f.smoothing_params().unwrap();
        assert!((params.0 - 0.6).abs() < 1e-12);
        assert!((params.1.unwrap() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_smoothing_params_ma_returns_zero_alpha() {
        let data = make_constant(5, 3.0);
        let mut f = Forecaster::new(SmoothingMethod::MovingAverage { window: 3 });
        f.fit(&data).unwrap();
        let params = f.smoothing_params().unwrap();
        assert_eq!(params.0, 0.0);
        assert!(params.1.is_none());
    }

    // -----------------------------------------------------------------------
    // DataPoint helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_data_point_new() {
        let dp = DataPoint::new(12345, 2.71);
        assert_eq!(dp.timestamp, 12345);
        assert!((dp.value - 2.71).abs() < 1e-12);
    }
}
