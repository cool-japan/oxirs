//! Time series gap detection and interpolation.
//!
//! Identifies gaps (consecutive missing samples exceeding a threshold) in
//! irregularly sampled time series and fills them using various interpolation
//! methods to produce a uniform-step grid.

/// A detected gap in a time series.
#[derive(Debug, Clone, PartialEq)]
pub struct Gap {
    /// Timestamp of the last sample *before* the gap (inclusive).
    pub start_ms: i64,
    /// Timestamp of the first sample *after* the gap (inclusive).
    pub end_ms: i64,
    /// Total gap duration = `end_ms - start_ms`.
    pub duration_ms: i64,
}

/// A single time-series data point.
#[derive(Debug, Clone, PartialEq)]
pub struct DataPoint {
    /// Unix epoch timestamp in milliseconds.
    pub timestamp: i64,
    /// Observed value.
    pub value: f64,
}

impl DataPoint {
    /// Create a new data point.
    pub fn new(timestamp: i64, value: f64) -> Self {
        Self { timestamp, value }
    }
}

/// Interpolation strategy used when filling gaps.
#[derive(Debug, Clone, PartialEq)]
pub enum InterpolationMethod {
    /// Linear interpolation between the two bounding samples.
    Linear,
    /// Step function — carry forward the previous known value.
    StepPrevious,
    /// Step function — use the next known value.
    StepNext,
    /// Fill every missing point with a fixed constant.
    Constant(f64),
    /// Cubic spline interpolation (falls back to linear when < 4 points).
    Spline,
}

/// Stateless helper functions for gap detection and interpolation.
pub struct GapFiller;

impl GapFiller {
    /// Detect all gaps in `data` where consecutive sample spacing exceeds
    /// `max_gap_ms` milliseconds.
    ///
    /// `data` must be sorted by ascending timestamp.  Returns an empty `Vec`
    /// if `data` has fewer than 2 points.
    pub fn detect_gaps(data: &[DataPoint], max_gap_ms: i64) -> Vec<Gap> {
        if data.len() < 2 {
            return Vec::new();
        }
        let mut gaps = Vec::new();
        for window in data.windows(2) {
            let diff = window[1].timestamp - window[0].timestamp;
            if diff > max_gap_ms {
                gaps.push(Gap {
                    start_ms: window[0].timestamp,
                    end_ms: window[1].timestamp,
                    duration_ms: diff,
                });
            }
        }
        gaps
    }

    /// Linearly interpolate the value at `target_ms` given the surrounding
    /// data.  Returns `None` if `target_ms` is outside the range of `data`,
    /// or if `data` is empty.
    pub fn fill_linear(data: &[DataPoint], target_ms: i64) -> Option<f64> {
        if data.is_empty() {
            return None;
        }
        // Exact match
        if let Some(pt) = data.iter().find(|p| p.timestamp == target_ms) {
            return Some(pt.value);
        }
        // Find bounding points
        let (before, after) = Self::bounding_points(data, target_ms)?;
        let span = (after.timestamp - before.timestamp) as f64;
        if span == 0.0 {
            return Some(before.value);
        }
        let t = (target_ms - before.timestamp) as f64 / span;
        Some(before.value + t * (after.value - before.value))
    }

    /// Step-previous interpolation — use the last known value before `target_ms`.
    pub fn fill_step_previous(data: &[DataPoint], target_ms: i64) -> Option<f64> {
        if data.is_empty() {
            return None;
        }
        data.iter()
            .filter(|p| p.timestamp <= target_ms)
            .last()
            .map(|p| p.value)
    }

    /// Step-next interpolation — use the first known value after (or at) `target_ms`.
    pub fn fill_step_next(data: &[DataPoint], target_ms: i64) -> Option<f64> {
        if data.is_empty() {
            return None;
        }
        data.iter()
            .find(|p| p.timestamp >= target_ms)
            .map(|p| p.value)
    }

    /// Interpolate at `target_ms` using the given method.
    pub fn interpolate_at(
        data: &[DataPoint],
        target_ms: i64,
        method: &InterpolationMethod,
    ) -> Option<f64> {
        match method {
            InterpolationMethod::Linear => Self::fill_linear(data, target_ms),
            InterpolationMethod::StepPrevious => Self::fill_step_previous(data, target_ms),
            InterpolationMethod::StepNext => Self::fill_step_next(data, target_ms),
            InterpolationMethod::Constant(c) => Some(*c),
            InterpolationMethod::Spline => Self::fill_spline(data, target_ms),
        }
    }

    /// Produce a uniform grid of data points from the first to the last
    /// timestamp in `data` with spacing `step_ms`, filling any missing
    /// positions using `method`.
    ///
    /// Returns an empty `Vec` for empty input.
    pub fn fill_gaps(
        data: &[DataPoint],
        step_ms: i64,
        method: InterpolationMethod,
    ) -> Vec<DataPoint> {
        if data.is_empty() || step_ms <= 0 {
            return Vec::new();
        }
        let first = data.first().expect("checked non-empty").timestamp;
        let last = data.last().expect("checked non-empty").timestamp;

        let mut result = Vec::new();
        let mut t = first;
        while t <= last {
            let value = if let Some(pt) = data.iter().find(|p| p.timestamp == t) {
                pt.value
            } else {
                Self::interpolate_at(data, t, &method).unwrap_or(f64::NAN)
            };
            result.push(DataPoint::new(t, value));
            t += step_ms;
        }
        result
    }

    /// Compute the data coverage percentage over the window
    /// `[start_ms, end_ms)` given an expected sample interval `expected_step_ms`.
    ///
    /// Returns a value in `[0.0, 100.0]`.
    pub fn coverage_pct(
        data: &[DataPoint],
        start_ms: i64,
        end_ms: i64,
        expected_step_ms: i64,
    ) -> f64 {
        if expected_step_ms <= 0 || end_ms <= start_ms {
            return 0.0;
        }
        let expected_count =
            ((end_ms - start_ms) as f64 / expected_step_ms as f64).ceil() as usize;
        if expected_count == 0 {
            return 100.0;
        }
        let actual_count = data
            .iter()
            .filter(|p| p.timestamp >= start_ms && p.timestamp < end_ms)
            .count();
        (actual_count as f64 / expected_count as f64 * 100.0).min(100.0)
    }

    /// Return the longest single gap, or `None` if there are no gaps.
    pub fn longest_gap(data: &[DataPoint]) -> Option<Gap> {
        if data.len() < 2 {
            return None;
        }
        data.windows(2)
            .map(|w| Gap {
                start_ms: w[0].timestamp,
                end_ms: w[1].timestamp,
                duration_ms: w[1].timestamp - w[0].timestamp,
            })
            .max_by_key(|g| g.duration_ms)
    }

    /// Sum the duration of all provided gaps.
    pub fn total_gap_duration(gaps: &[Gap]) -> i64 {
        gaps.iter().map(|g| g.duration_ms).sum()
    }

    // --- private helpers ---

    /// Find the last point with `timestamp <= target_ms` and the first with
    /// `timestamp >= target_ms`.  Returns `None` if either bound is absent.
    fn bounding_points(data: &[DataPoint], target_ms: i64) -> Option<(&DataPoint, &DataPoint)> {
        let before = data.iter().filter(|p| p.timestamp <= target_ms).last()?;
        let after = data.iter().find(|p| p.timestamp >= target_ms)?;
        Some((before, after))
    }

    /// Cubic spline interpolation using a natural spline.
    ///
    /// Falls back to linear interpolation when there are fewer than 4 points.
    fn fill_spline(data: &[DataPoint], target_ms: i64) -> Option<f64> {
        if data.len() < 4 {
            return Self::fill_linear(data, target_ms);
        }
        // Find segment
        let (before, after) = Self::bounding_points(data, target_ms)?;
        if before.timestamp == after.timestamp {
            return Some(before.value);
        }

        // For a production spline we would solve the tridiagonal system; here we
        // use a simple cubic Hermite spline (Catmull-Rom) between the two
        // bounding points using their neighbours as tangent guides.
        let idx_before = data.iter().position(|p| p.timestamp == before.timestamp)?;
        let idx_after = data.iter().position(|p| p.timestamp == after.timestamp)?;

        let p0 = if idx_before > 0 {
            &data[idx_before - 1]
        } else {
            before
        };
        let p3 = if idx_after + 1 < data.len() {
            &data[idx_after + 1]
        } else {
            after
        };

        let t = (target_ms - before.timestamp) as f64
            / (after.timestamp - before.timestamp) as f64;

        // Catmull-Rom coefficients
        let t2 = t * t;
        let t3 = t2 * t;
        let v = 0.5
            * ((2.0 * before.value)
                + (-p0.value + after.value) * t
                + (2.0 * p0.value - 5.0 * before.value + 4.0 * after.value - p3.value) * t2
                + (-p0.value + 3.0 * before.value - 3.0 * after.value + p3.value) * t3);
        Some(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pts(pairs: &[(i64, f64)]) -> Vec<DataPoint> {
        pairs.iter().map(|(t, v)| DataPoint::new(*t, *v)).collect()
    }

    // --- detect_gaps ---
    #[test]
    fn test_detect_gaps_empty() {
        assert!(GapFiller::detect_gaps(&[], 1000).is_empty());
    }

    #[test]
    fn test_detect_gaps_single_point() {
        let data = pts(&[(0, 1.0)]);
        assert!(GapFiller::detect_gaps(&data, 1000).is_empty());
    }

    #[test]
    fn test_detect_gaps_no_gaps() {
        let data = pts(&[(0, 1.0), (1000, 2.0), (2000, 3.0)]);
        assert!(GapFiller::detect_gaps(&data, 1001).is_empty());
    }

    #[test]
    fn test_detect_gaps_one_gap() {
        let data = pts(&[(0, 1.0), (5000, 2.0)]);
        let gaps = GapFiller::detect_gaps(&data, 1000);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].duration_ms, 5000);
    }

    #[test]
    fn test_detect_gaps_multiple() {
        let data = pts(&[(0, 0.0), (1000, 1.0), (5000, 2.0), (6000, 3.0), (10000, 4.0)]);
        let gaps = GapFiller::detect_gaps(&data, 1500);
        assert_eq!(gaps.len(), 2);
    }

    #[test]
    fn test_detect_gaps_start_end() {
        let data = pts(&[(100, 1.0), (5000, 2.0)]);
        let gaps = GapFiller::detect_gaps(&data, 1000);
        assert_eq!(gaps[0].start_ms, 100);
        assert_eq!(gaps[0].end_ms, 5000);
    }

    // --- fill_linear ---
    #[test]
    fn test_fill_linear_empty() {
        assert!(GapFiller::fill_linear(&[], 0).is_none());
    }

    #[test]
    fn test_fill_linear_exact_match() {
        let data = pts(&[(0, 10.0), (1000, 20.0)]);
        assert_eq!(GapFiller::fill_linear(&data, 0), Some(10.0));
        assert_eq!(GapFiller::fill_linear(&data, 1000), Some(20.0));
    }

    #[test]
    fn test_fill_linear_midpoint() {
        let data = pts(&[(0, 0.0), (1000, 100.0)]);
        let v = GapFiller::fill_linear(&data, 500).unwrap();
        assert!((v - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_fill_linear_outside_range() {
        let data = pts(&[(0, 0.0), (1000, 100.0)]);
        assert!(GapFiller::fill_linear(&data, 2000).is_none());
    }

    #[test]
    fn test_fill_linear_quarter_point() {
        let data = pts(&[(0, 0.0), (1000, 40.0)]);
        let v = GapFiller::fill_linear(&data, 250).unwrap();
        assert!((v - 10.0).abs() < 1e-9);
    }

    // --- fill_step_previous ---
    #[test]
    fn test_fill_step_previous_empty() {
        assert!(GapFiller::fill_step_previous(&[], 0).is_none());
    }

    #[test]
    fn test_fill_step_previous_basic() {
        let data = pts(&[(0, 1.0), (1000, 2.0), (2000, 3.0)]);
        assert_eq!(GapFiller::fill_step_previous(&data, 1500), Some(2.0));
    }

    #[test]
    fn test_fill_step_previous_exact() {
        let data = pts(&[(0, 5.0), (1000, 10.0)]);
        assert_eq!(GapFiller::fill_step_previous(&data, 1000), Some(10.0));
    }

    #[test]
    fn test_fill_step_previous_before_first() {
        let data = pts(&[(1000, 5.0)]);
        assert!(GapFiller::fill_step_previous(&data, 500).is_none());
    }

    // --- fill_step_next ---
    #[test]
    fn test_fill_step_next_empty() {
        assert!(GapFiller::fill_step_next(&[], 0).is_none());
    }

    #[test]
    fn test_fill_step_next_basic() {
        let data = pts(&[(0, 1.0), (1000, 2.0), (2000, 3.0)]);
        assert_eq!(GapFiller::fill_step_next(&data, 500), Some(2.0));
    }

    #[test]
    fn test_fill_step_next_exact() {
        let data = pts(&[(0, 5.0), (1000, 10.0)]);
        assert_eq!(GapFiller::fill_step_next(&data, 0), Some(5.0));
    }

    #[test]
    fn test_fill_step_next_after_last() {
        let data = pts(&[(0, 5.0)]);
        assert!(GapFiller::fill_step_next(&data, 1000).is_none());
    }

    // --- interpolate_at ---
    #[test]
    fn test_interpolate_at_constant() {
        let data = pts(&[(0, 0.0), (1000, 100.0)]);
        let v = GapFiller::interpolate_at(&data, 500, &InterpolationMethod::Constant(42.0));
        assert_eq!(v, Some(42.0));
    }

    #[test]
    fn test_interpolate_at_linear() {
        let data = pts(&[(0, 0.0), (1000, 100.0)]);
        let v = GapFiller::interpolate_at(&data, 500, &InterpolationMethod::Linear).unwrap();
        assert!((v - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_interpolate_at_step_previous() {
        let data = pts(&[(0, 1.0), (1000, 2.0)]);
        let v =
            GapFiller::interpolate_at(&data, 500, &InterpolationMethod::StepPrevious).unwrap();
        assert_eq!(v, 1.0);
    }

    #[test]
    fn test_interpolate_at_step_next() {
        let data = pts(&[(0, 1.0), (1000, 2.0)]);
        let v = GapFiller::interpolate_at(&data, 500, &InterpolationMethod::StepNext).unwrap();
        assert_eq!(v, 2.0);
    }

    // --- fill_gaps ---
    #[test]
    fn test_fill_gaps_empty() {
        assert!(GapFiller::fill_gaps(&[], 1000, InterpolationMethod::Linear).is_empty());
    }

    #[test]
    fn test_fill_gaps_no_gaps_identity() {
        let data = pts(&[(0, 1.0), (1000, 2.0), (2000, 3.0)]);
        let result = GapFiller::fill_gaps(&data, 1000, InterpolationMethod::Linear);
        assert_eq!(result.len(), 3);
        assert!((result[0].value - 1.0).abs() < 1e-9);
        assert!((result[1].value - 2.0).abs() < 1e-9);
        assert!((result[2].value - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_fill_gaps_inserts_interpolated_points() {
        let data = pts(&[(0, 0.0), (2000, 20.0)]);
        let result = GapFiller::fill_gaps(&data, 1000, InterpolationMethod::Linear);
        assert_eq!(result.len(), 3); // 0, 1000, 2000
        assert!((result[1].value - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_fill_gaps_constant_method() {
        let data = pts(&[(0, 5.0), (2000, 5.0)]);
        let result = GapFiller::fill_gaps(&data, 1000, InterpolationMethod::Constant(99.0));
        assert_eq!(result[1].value, 99.0);
    }

    #[test]
    fn test_fill_gaps_step_previous() {
        let data = pts(&[(0, 10.0), (3000, 30.0)]);
        let result = GapFiller::fill_gaps(&data, 1000, InterpolationMethod::StepPrevious);
        // result[1].value should be 10.0 (step-previous of position 1000)
        assert!((result[1].value - 10.0).abs() < 1e-9);
    }

    // --- coverage_pct ---
    #[test]
    fn test_coverage_pct_full() {
        let data = pts(&[(0, 1.0), (1000, 2.0), (2000, 3.0), (3000, 4.0), (4000, 5.0)]);
        let pct = GapFiller::coverage_pct(&data, 0, 5000, 1000);
        assert!((pct - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_coverage_pct_zero_step() {
        let data = pts(&[(0, 1.0)]);
        assert_eq!(GapFiller::coverage_pct(&data, 0, 1000, 0), 0.0);
    }

    #[test]
    fn test_coverage_pct_partial() {
        // 5 points in [0, 10000), expected 10 at step 1000 → 50%
        let data = pts(&[(0, 1.0), (1000, 2.0), (2000, 3.0), (3000, 4.0), (4000, 5.0)]);
        let pct = GapFiller::coverage_pct(&data, 0, 10000, 1000);
        assert!((pct - 50.0).abs() < 1e-6);
    }

    // --- longest_gap ---
    #[test]
    fn test_longest_gap_empty() {
        assert!(GapFiller::longest_gap(&[]).is_none());
    }

    #[test]
    fn test_longest_gap_single() {
        let data = pts(&[(0, 0.0)]);
        assert!(GapFiller::longest_gap(&data).is_none());
    }

    #[test]
    fn test_longest_gap_basic() {
        let data = pts(&[(0, 0.0), (1000, 1.0), (5000, 2.0), (6000, 3.0)]);
        let g = GapFiller::longest_gap(&data).unwrap();
        assert_eq!(g.duration_ms, 4000);
    }

    #[test]
    fn test_longest_gap_uniform() {
        let data = pts(&[(0, 0.0), (1000, 1.0), (2000, 2.0)]);
        let g = GapFiller::longest_gap(&data).unwrap();
        assert_eq!(g.duration_ms, 1000);
    }

    // --- total_gap_duration ---
    #[test]
    fn test_total_gap_duration_empty() {
        assert_eq!(GapFiller::total_gap_duration(&[]), 0);
    }

    #[test]
    fn test_total_gap_duration_basic() {
        let gaps = vec![
            Gap { start_ms: 0, end_ms: 2000, duration_ms: 2000 },
            Gap { start_ms: 5000, end_ms: 8000, duration_ms: 3000 },
        ];
        assert_eq!(GapFiller::total_gap_duration(&gaps), 5000);
    }

    // --- DataPoint helpers ---
    #[test]
    fn test_data_point_new() {
        let p = DataPoint::new(42, 2.71);
        assert_eq!(p.timestamp, 42);
        assert!((p.value - 2.71).abs() < 1e-9);
    }

    // --- spline fallback ---
    #[test]
    fn test_spline_fallback_to_linear_few_points() {
        let data = pts(&[(0, 0.0), (1000, 100.0)]);
        let v = GapFiller::interpolate_at(&data, 500, &InterpolationMethod::Spline).unwrap();
        assert!((v - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_spline_with_enough_points() {
        let data = pts(&[(0, 0.0), (1000, 1.0), (2000, 4.0), (3000, 9.0)]);
        // Should not panic
        let v = GapFiller::interpolate_at(&data, 1500, &InterpolationMethod::Spline);
        assert!(v.is_some());
    }
}
