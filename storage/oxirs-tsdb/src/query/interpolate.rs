//! Interpolation methods for filling missing data
//!
//! Provides various interpolation strategies:
//! - Linear interpolation
//! - Forward fill (LOCF - Last Observation Carried Forward)
//! - Backward fill
//! - Spline interpolation

use crate::error::{TsdbError, TsdbResult};
use crate::series::DataPoint;
use chrono::{DateTime, Duration, Utc};

/// Interpolation method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolateMethod {
    /// Linear interpolation between adjacent points
    Linear,
    /// Forward fill (carry last observation forward)
    ForwardFill,
    /// Backward fill (use next value)
    BackwardFill,
    /// Nearest value (use closest point)
    Nearest,
    /// Constant value fill
    Constant(f64),
    /// Step function (hold until next point)
    Step,
}

/// Interpolator for filling missing values
pub struct Interpolator {
    method: InterpolateMethod,
    /// Maximum gap to fill (None = unlimited)
    max_gap: Option<Duration>,
}

impl Interpolator {
    /// Create a new interpolator
    pub fn new(method: InterpolateMethod) -> Self {
        Self {
            method,
            max_gap: None,
        }
    }

    /// Set maximum gap to interpolate across
    pub fn with_max_gap(mut self, max_gap: Duration) -> Self {
        self.max_gap = Some(max_gap);
        self
    }

    /// Interpolate a value at a specific timestamp
    pub fn interpolate_at(
        &self,
        timestamp: DateTime<Utc>,
        points: &[DataPoint],
    ) -> TsdbResult<f64> {
        if points.is_empty() {
            return Err(TsdbError::Query(
                "No data points for interpolation".to_string(),
            ));
        }

        // Find surrounding points
        let (before, after) = find_surrounding_points(timestamp, points);

        match self.method {
            InterpolateMethod::Linear => self.linear_interpolate(timestamp, before, after),
            InterpolateMethod::ForwardFill => before
                .map(|p| p.value)
                .ok_or_else(|| TsdbError::Query("No previous value for forward fill".to_string())),
            InterpolateMethod::BackwardFill => after
                .map(|p| p.value)
                .ok_or_else(|| TsdbError::Query("No next value for backward fill".to_string())),
            InterpolateMethod::Nearest => match (before, after) {
                (Some(b), Some(a)) => {
                    let gap_before = (timestamp - b.timestamp).num_milliseconds().abs();
                    let gap_after = (a.timestamp - timestamp).num_milliseconds().abs();
                    if gap_before <= gap_after {
                        Ok(b.value)
                    } else {
                        Ok(a.value)
                    }
                }
                (Some(b), None) => Ok(b.value),
                (None, Some(a)) => Ok(a.value),
                (None, None) => Err(TsdbError::Query("No surrounding points".to_string())),
            },
            InterpolateMethod::Constant(v) => Ok(v),
            InterpolateMethod::Step => before.map(|p| p.value).ok_or_else(|| {
                TsdbError::Query("No previous value for step interpolation".to_string())
            }),
        }
    }

    /// Linear interpolation between two points
    fn linear_interpolate(
        &self,
        timestamp: DateTime<Utc>,
        before: Option<DataPoint>,
        after: Option<DataPoint>,
    ) -> TsdbResult<f64> {
        match (before, after) {
            (Some(b), Some(a)) => {
                // Check max gap
                if let Some(max_gap) = self.max_gap {
                    if (a.timestamp - b.timestamp) > max_gap {
                        return Err(TsdbError::Query(
                            "Gap too large for interpolation".to_string(),
                        ));
                    }
                }

                let total_duration = (a.timestamp - b.timestamp).num_milliseconds() as f64;
                let elapsed = (timestamp - b.timestamp).num_milliseconds() as f64;

                if total_duration == 0.0 {
                    return Ok(b.value);
                }

                let ratio = elapsed / total_duration;
                Ok(b.value + ratio * (a.value - b.value))
            }
            (Some(b), None) => Ok(b.value), // Extrapolate using last known value
            (None, Some(a)) => Ok(a.value), // Extrapolate using first known value
            (None, None) => Err(TsdbError::Query(
                "No surrounding points for interpolation".to_string(),
            )),
        }
    }

    /// Fill gaps in a time series at regular intervals
    pub fn fill_at_interval(
        &self,
        points: &[DataPoint],
        interval: Duration,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> TsdbResult<Vec<DataPoint>> {
        if points.is_empty() {
            return Ok(Vec::new());
        }

        let actual_start = start.unwrap_or_else(|| points.first().unwrap().timestamp);
        let actual_end = end.unwrap_or_else(|| points.last().unwrap().timestamp);

        let mut result = Vec::new();
        let mut current = actual_start;

        while current <= actual_end {
            // Check if we have an exact match
            if let Some(exact) = points.iter().find(|p| {
                (p.timestamp - current).num_milliseconds().abs() < interval.num_milliseconds() / 2
            }) {
                result.push(*exact);
            } else {
                // Interpolate
                match self.interpolate_at(current, points) {
                    Ok(value) => {
                        result.push(DataPoint {
                            timestamp: current,
                            value,
                        });
                    }
                    Err(_) => {
                        // Skip if cannot interpolate (e.g., gap too large)
                    }
                }
            }

            current += interval;
        }

        Ok(result)
    }

    /// Upsample data points to higher frequency
    pub fn upsample(
        &self,
        points: &[DataPoint],
        target_interval: Duration,
    ) -> TsdbResult<Vec<DataPoint>> {
        if points.len() < 2 {
            return Ok(points.to_vec());
        }

        let start = points.first().unwrap().timestamp;
        let end = points.last().unwrap().timestamp;

        self.fill_at_interval(points, target_interval, Some(start), Some(end))
    }
}

/// Find the nearest points before and after a given timestamp
fn find_surrounding_points(
    timestamp: DateTime<Utc>,
    points: &[DataPoint],
) -> (Option<DataPoint>, Option<DataPoint>) {
    let mut before: Option<DataPoint> = None;
    let mut after: Option<DataPoint> = None;

    for point in points {
        if point.timestamp <= timestamp {
            before = Some(*point);
        } else {
            after = Some(*point);
            break;
        }
    }

    (before, after)
}

/// Convenience function for linear interpolation
pub fn interpolate_linear(timestamp: DateTime<Utc>, points: &[DataPoint]) -> TsdbResult<f64> {
    Interpolator::new(InterpolateMethod::Linear).interpolate_at(timestamp, points)
}

/// Convenience function for forward fill
pub fn forward_fill(timestamp: DateTime<Utc>, points: &[DataPoint]) -> TsdbResult<f64> {
    Interpolator::new(InterpolateMethod::ForwardFill).interpolate_at(timestamp, points)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_points() -> Vec<DataPoint> {
        let start = Utc::now();
        vec![
            DataPoint {
                timestamp: start,
                value: 10.0,
            },
            DataPoint {
                timestamp: start + Duration::seconds(10),
                value: 20.0,
            },
            DataPoint {
                timestamp: start + Duration::seconds(20),
                value: 30.0,
            },
        ]
    }

    #[test]
    fn test_linear_interpolation() {
        let points = create_test_points();
        let mid_time = points[0].timestamp + Duration::seconds(5);

        let result = interpolate_linear(mid_time, &points).unwrap();

        // At 5 seconds, should be 15.0 (linear between 10 and 20)
        assert!((result - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_forward_fill() {
        let points = create_test_points();
        let mid_time = points[0].timestamp + Duration::seconds(5);

        let result = forward_fill(mid_time, &points).unwrap();

        // Forward fill uses previous value: 10.0
        assert!((result - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_backward_fill() {
        let points = create_test_points();
        let mid_time = points[0].timestamp + Duration::seconds(5);

        let interp = Interpolator::new(InterpolateMethod::BackwardFill);
        let result = interp.interpolate_at(mid_time, &points).unwrap();

        // Backward fill uses next value: 20.0
        assert!((result - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_nearest() {
        let points = create_test_points();

        // At 3 seconds, nearest is first point (10.0)
        let near_first = points[0].timestamp + Duration::seconds(3);
        let interp = Interpolator::new(InterpolateMethod::Nearest);
        let result1 = interp.interpolate_at(near_first, &points).unwrap();
        assert!((result1 - 10.0).abs() < 0.001);

        // At 7 seconds, nearest is second point (20.0)
        let near_second = points[0].timestamp + Duration::seconds(7);
        let result2 = interp.interpolate_at(near_second, &points).unwrap();
        assert!((result2 - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_constant_fill() {
        let points = create_test_points();
        let mid_time = points[0].timestamp + Duration::seconds(5);

        let interp = Interpolator::new(InterpolateMethod::Constant(42.0));
        let result = interp.interpolate_at(mid_time, &points).unwrap();

        assert!((result - 42.0).abs() < 0.001);
    }

    #[test]
    fn test_fill_at_interval() {
        let start = Utc::now();
        let points = vec![
            DataPoint {
                timestamp: start,
                value: 0.0,
            },
            DataPoint {
                timestamp: start + Duration::seconds(10),
                value: 100.0,
            },
        ];

        let interp = Interpolator::new(InterpolateMethod::Linear);
        let filled = interp
            .fill_at_interval(&points, Duration::seconds(2), None, None)
            .unwrap();

        // Should have 6 points: 0, 2, 4, 6, 8, 10 seconds
        assert_eq!(filled.len(), 6);

        // Check linear progression
        for (i, point) in filled.iter().enumerate() {
            let expected = i as f64 * 20.0; // 0, 20, 40, 60, 80, 100
            assert!(
                (point.value - expected).abs() < 1.0,
                "Point {} expected {}, got {}",
                i,
                expected,
                point.value
            );
        }
    }

    #[test]
    fn test_max_gap_limit() {
        let start = Utc::now();
        let points = vec![
            DataPoint {
                timestamp: start,
                value: 0.0,
            },
            DataPoint {
                timestamp: start + Duration::hours(1),
                value: 100.0,
            },
        ];

        let interp =
            Interpolator::new(InterpolateMethod::Linear).with_max_gap(Duration::minutes(10));

        let mid_time = start + Duration::minutes(30);
        let result = interp.interpolate_at(mid_time, &points);

        // Should fail because gap is 1 hour, max is 10 minutes
        assert!(result.is_err());
    }

    #[test]
    fn test_upsample() {
        let start = Utc::now();
        let points = vec![
            DataPoint {
                timestamp: start,
                value: 0.0,
            },
            DataPoint {
                timestamp: start + Duration::seconds(4),
                value: 40.0,
            },
        ];

        let interp = Interpolator::new(InterpolateMethod::Linear);
        let upsampled = interp.upsample(&points, Duration::seconds(1)).unwrap();

        // Should have 5 points: 0, 1, 2, 3, 4 seconds
        assert_eq!(upsampled.len(), 5);
        assert!((upsampled[2].value - 20.0).abs() < 0.001);
    }
}
