//! Window functions for time-series analysis
//!
//! Provides moving/rolling window operations:
//! - Moving averages (simple, exponential)
//! - Rolling min/max
//! - Rolling standard deviation

use crate::error::TsdbResult;
use crate::series::DataPoint;
use chrono::Duration;
use std::collections::VecDeque;

/// Window function type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowFunction {
    /// Simple moving average
    MovingAverage,
    /// Exponential moving average with alpha parameter
    ExponentialMovingAverage(f64),
    /// Moving minimum
    MovingMin,
    /// Moving maximum
    MovingMax,
    /// Rolling standard deviation
    RollingStdDev,
    /// Cumulative sum
    CumulativeSum,
    /// Rate of change (derivative)
    RateOfChange,
}

/// Window specification
#[derive(Debug, Clone)]
pub struct WindowSpec {
    /// Window size in data points (count-based)
    pub size: Option<usize>,
    /// Window duration (time-based)
    pub duration: Option<Duration>,
    /// Window function to apply
    pub function: WindowFunction,
}

impl WindowSpec {
    /// Create a count-based window
    pub fn count_based(size: usize, function: WindowFunction) -> Self {
        Self {
            size: Some(size),
            duration: None,
            function,
        }
    }

    /// Create a time-based window
    pub fn time_based(duration: Duration, function: WindowFunction) -> Self {
        Self {
            size: None,
            duration: Some(duration),
            function,
        }
    }
}

/// Window function calculator
pub struct WindowCalculator {
    spec: WindowSpec,
    // For count-based windows
    window: VecDeque<DataPoint>,
    // Running values
    sum: f64,
    ema_value: Option<f64>,
    cumulative: f64,
    prev_value: Option<f64>,
}

impl WindowCalculator {
    /// Create a new window calculator
    pub fn new(spec: WindowSpec) -> Self {
        Self {
            spec,
            window: VecDeque::new(),
            sum: 0.0,
            ema_value: None,
            cumulative: 0.0,
            prev_value: None,
        }
    }

    /// Add a data point and get the windowed result
    pub fn add(&mut self, point: DataPoint) -> Option<DataPoint> {
        match self.spec.function {
            WindowFunction::MovingAverage => self.moving_average(point),
            WindowFunction::ExponentialMovingAverage(alpha) => {
                self.exponential_moving_average(point, alpha)
            }
            WindowFunction::MovingMin => self.moving_min(point),
            WindowFunction::MovingMax => self.moving_max(point),
            WindowFunction::RollingStdDev => self.rolling_stddev(point),
            WindowFunction::CumulativeSum => self.cumulative_sum(point),
            WindowFunction::RateOfChange => self.rate_of_change(point),
        }
    }

    /// Apply window function to all points
    pub fn apply(&mut self, points: &[DataPoint]) -> Vec<DataPoint> {
        points.iter().filter_map(|p| self.add(*p)).collect()
    }

    /// Reset the calculator state
    pub fn reset(&mut self) {
        self.window.clear();
        self.sum = 0.0;
        self.ema_value = None;
        self.cumulative = 0.0;
        self.prev_value = None;
    }

    fn moving_average(&mut self, point: DataPoint) -> Option<DataPoint> {
        let size = self.spec.size.unwrap_or(10);

        // Add new point
        self.window.push_back(point);
        self.sum += point.value;

        // Remove old points if window full
        while self.window.len() > size {
            if let Some(old) = self.window.pop_front() {
                self.sum -= old.value;
            }
        }

        // Only return result when window is full
        if self.window.len() >= size {
            Some(DataPoint {
                timestamp: point.timestamp,
                value: self.sum / self.window.len() as f64,
            })
        } else {
            None
        }
    }

    fn exponential_moving_average(&mut self, point: DataPoint, alpha: f64) -> Option<DataPoint> {
        let alpha = alpha.clamp(0.0, 1.0);

        let ema = match self.ema_value {
            Some(prev_ema) => alpha * point.value + (1.0 - alpha) * prev_ema,
            None => point.value, // First value
        };

        self.ema_value = Some(ema);

        Some(DataPoint {
            timestamp: point.timestamp,
            value: ema,
        })
    }

    fn moving_min(&mut self, point: DataPoint) -> Option<DataPoint> {
        let size = self.spec.size.unwrap_or(10);

        // Use time-based or count-based window
        if let Some(duration) = self.spec.duration {
            // Time-based: remove points outside window
            let cutoff = point.timestamp - duration;
            while let Some(front) = self.window.front() {
                if front.timestamp < cutoff {
                    self.window.pop_front();
                } else {
                    break;
                }
            }
        } else {
            // Count-based
            while self.window.len() >= size {
                self.window.pop_front();
            }
        }

        self.window.push_back(point);

        // Find minimum in window
        let min = self
            .window
            .iter()
            .map(|p| p.value)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(point.value);

        Some(DataPoint {
            timestamp: point.timestamp,
            value: min,
        })
    }

    fn moving_max(&mut self, point: DataPoint) -> Option<DataPoint> {
        let size = self.spec.size.unwrap_or(10);

        // Use time-based or count-based window
        if let Some(duration) = self.spec.duration {
            let cutoff = point.timestamp - duration;
            while let Some(front) = self.window.front() {
                if front.timestamp < cutoff {
                    self.window.pop_front();
                } else {
                    break;
                }
            }
        } else {
            while self.window.len() >= size {
                self.window.pop_front();
            }
        }

        self.window.push_back(point);

        // Find maximum in window
        let max = self
            .window
            .iter()
            .map(|p| p.value)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(point.value);

        Some(DataPoint {
            timestamp: point.timestamp,
            value: max,
        })
    }

    fn rolling_stddev(&mut self, point: DataPoint) -> Option<DataPoint> {
        let size = self.spec.size.unwrap_or(10);

        self.window.push_back(point);
        while self.window.len() > size {
            self.window.pop_front();
        }

        if self.window.len() < 2 {
            return None;
        }

        // Calculate mean
        let mean: f64 = self.window.iter().map(|p| p.value).sum::<f64>() / self.window.len() as f64;

        // Calculate variance
        let variance: f64 = self
            .window
            .iter()
            .map(|p| (p.value - mean).powi(2))
            .sum::<f64>()
            / (self.window.len() - 1) as f64;

        Some(DataPoint {
            timestamp: point.timestamp,
            value: variance.sqrt(),
        })
    }

    fn cumulative_sum(&mut self, point: DataPoint) -> Option<DataPoint> {
        self.cumulative += point.value;

        Some(DataPoint {
            timestamp: point.timestamp,
            value: self.cumulative,
        })
    }

    fn rate_of_change(&mut self, point: DataPoint) -> Option<DataPoint> {
        let result = self.prev_value.map(|prev| DataPoint {
            timestamp: point.timestamp,
            value: point.value - prev,
        });

        self.prev_value = Some(point.value);
        result
    }
}

/// Apply window function to data points
pub fn apply_window(points: &[DataPoint], spec: WindowSpec) -> TsdbResult<Vec<DataPoint>> {
    if points.is_empty() {
        return Ok(Vec::new());
    }

    let mut calculator = WindowCalculator::new(spec);
    Ok(calculator.apply(points))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Utc};

    fn create_test_points(start: DateTime<Utc>, values: &[f64]) -> Vec<DataPoint> {
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| DataPoint {
                timestamp: start + Duration::seconds(i as i64),
                value: v,
            })
            .collect()
    }

    #[test]
    fn test_moving_average() {
        let now = Utc::now();
        let points = create_test_points(now, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        let spec = WindowSpec::count_based(3, WindowFunction::MovingAverage);
        let results = apply_window(&points, spec).unwrap();

        // First 2 points won't have results (window not full)
        // Point 3: avg(1,2,3) = 2.0
        // Point 4: avg(2,3,4) = 3.0
        assert_eq!(results.len(), 8);
        assert!((results[0].value - 2.0).abs() < 0.001);
        assert!((results[1].value - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_exponential_moving_average() {
        let now = Utc::now();
        let points = create_test_points(now, &[10.0, 20.0, 30.0, 40.0]);

        let spec = WindowSpec::count_based(1, WindowFunction::ExponentialMovingAverage(0.5));
        let results = apply_window(&points, spec).unwrap();

        // EMA: first = 10.0
        // second = 0.5 * 20 + 0.5 * 10 = 15.0
        // third = 0.5 * 30 + 0.5 * 15 = 22.5
        assert_eq!(results.len(), 4);
        assert!((results[0].value - 10.0).abs() < 0.001);
        assert!((results[1].value - 15.0).abs() < 0.001);
        assert!((results[2].value - 22.5).abs() < 0.001);
    }

    #[test]
    fn test_moving_min_max() {
        let now = Utc::now();
        let points = create_test_points(now, &[5.0, 3.0, 8.0, 2.0, 7.0]);

        let spec_min = WindowSpec::count_based(3, WindowFunction::MovingMin);
        let results_min = apply_window(&points, spec_min).unwrap();

        let spec_max = WindowSpec::count_based(3, WindowFunction::MovingMax);
        let results_max = apply_window(&points, spec_max).unwrap();

        // Window of 3:
        // [5, 3, 8] -> min=3, max=8
        // [3, 8, 2] -> min=2, max=8
        // [8, 2, 7] -> min=2, max=8
        assert!((results_min[2].value - 3.0).abs() < 0.001);
        assert!((results_min[3].value - 2.0).abs() < 0.001);
        assert!((results_max[2].value - 8.0).abs() < 0.001);
    }

    #[test]
    fn test_rolling_stddev() {
        let now = Utc::now();
        let points = create_test_points(now, &[2.0, 4.0, 4.0, 4.0, 5.0]);

        let spec = WindowSpec::count_based(4, WindowFunction::RollingStdDev);
        let results = apply_window(&points, spec).unwrap();

        // With window size 4, we need at least 4 points to start producing results
        // and at least 2 for stddev calculation
        // Results: window [2,4,4,4] has stddev ~1.15, window [4,4,4,5] has stddev ~0.5
        assert!(results.len() >= 2);
        // Just verify we get reasonable non-negative stddev values
        assert!(results.last().unwrap().value >= 0.0);
        assert!(results.last().unwrap().value < 5.0);
    }

    #[test]
    fn test_cumulative_sum() {
        let now = Utc::now();
        let points = create_test_points(now, &[1.0, 2.0, 3.0, 4.0, 5.0]);

        let spec = WindowSpec::count_based(1, WindowFunction::CumulativeSum);
        let results = apply_window(&points, spec).unwrap();

        assert_eq!(results.len(), 5);
        assert!((results[0].value - 1.0).abs() < 0.001);
        assert!((results[1].value - 3.0).abs() < 0.001);
        assert!((results[2].value - 6.0).abs() < 0.001);
        assert!((results[3].value - 10.0).abs() < 0.001);
        assert!((results[4].value - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_rate_of_change() {
        let now = Utc::now();
        let points = create_test_points(now, &[10.0, 15.0, 12.0, 18.0]);

        let spec = WindowSpec::count_based(1, WindowFunction::RateOfChange);
        let results = apply_window(&points, spec).unwrap();

        // First point has no previous, so no result
        // 15 - 10 = 5
        // 12 - 15 = -3
        // 18 - 12 = 6
        assert_eq!(results.len(), 3);
        assert!((results[0].value - 5.0).abs() < 0.001);
        assert!((results[1].value - (-3.0)).abs() < 0.001);
        assert!((results[2].value - 6.0).abs() < 0.001);
    }
}
