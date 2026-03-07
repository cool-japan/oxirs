//! Time-series rollup and downsampling engine (v1.1.0 round 16).
//!
//! Provides windowed aggregation (mean, sum, min, max, count, first, last) and
//! the Largest-Triangle-Three-Buckets (LTTB) visual downsampling algorithm.
//!
//! Reference: Steinarsson, S., "Downsampling Time Series for Visual
//! Representation", MSc Thesis, Reykjavik University, 2013.
//! <http://skemman.is/handle/1946/15343>

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregation function for a rollup window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RollupFn {
    /// Arithmetic mean of values in the window.
    Mean,
    /// Sum of values in the window.
    Sum,
    /// Minimum value in the window.
    Min,
    /// Maximum value in the window.
    Max,
    /// Count of data points in the window.
    Count,
    /// Last (most recent) value in the window.
    Last,
    /// First (earliest) value in the window.
    First,
}

/// A single time-series data point.
#[derive(Debug, Clone, PartialEq)]
pub struct DataPoint {
    /// Timestamp in milliseconds since an arbitrary epoch.
    pub timestamp_ms: u64,
    /// Observed value.
    pub value: f64,
}

impl DataPoint {
    /// Create a new `DataPoint`.
    pub fn new(timestamp_ms: u64, value: f64) -> Self {
        Self {
            timestamp_ms,
            value,
        }
    }
}

/// A rollup specification describing how to aggregate data.
#[derive(Debug, Clone)]
pub struct RollupSpec {
    /// Width of each aggregation window in milliseconds.
    pub window_ms: u64,
    /// Aggregation function to apply within each window.
    pub function: RollupFn,
}

impl RollupSpec {
    /// Create a new `RollupSpec`.
    pub fn new(window_ms: u64, function: RollupFn) -> Self {
        Self {
            window_ms,
            function,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// RollupEngine
// ──────────────────────────────────────────────────────────────────────────────

/// Engine that performs rollup/downsampling on time-series data.
pub struct RollupEngine;

impl RollupEngine {
    /// Create a new `RollupEngine`.
    pub fn new() -> Self {
        Self
    }

    /// Apply a single rollup spec to a slice of sorted data points.
    ///
    /// Points are grouped into non-overlapping windows of `spec.window_ms`.
    /// The window timestamp is the **start** of the window (i.e. the floor of
    /// the point timestamps to the nearest multiple of `window_ms`).
    /// Windows with no points are omitted from the output.
    pub fn rollup(&self, points: &[DataPoint], spec: &RollupSpec) -> Vec<DataPoint> {
        if points.is_empty() || spec.window_ms == 0 {
            return Vec::new();
        }

        let mut result: Vec<DataPoint> = Vec::new();
        let mut window_start: u64 = (points[0].timestamp_ms / spec.window_ms) * spec.window_ms;
        let mut window_points: Vec<f64> = Vec::new();

        let flush = |ws: u64, wp: &[f64], fn_: RollupFn| -> Option<DataPoint> {
            if wp.is_empty() {
                return None;
            }
            let value = Self::aggregate(wp, fn_);
            Some(DataPoint::new(ws, value))
        };

        for pt in points {
            let this_window = (pt.timestamp_ms / spec.window_ms) * spec.window_ms;
            if this_window != window_start {
                if let Some(dp) = flush(window_start, &window_points, spec.function) {
                    result.push(dp);
                }
                window_start = this_window;
                window_points.clear();
            }
            window_points.push(pt.value);
        }
        // Flush last window
        if let Some(dp) = flush(window_start, &window_points, spec.function) {
            result.push(dp);
        }

        result
    }

    /// Apply multiple rollup specs at different resolutions.
    ///
    /// Returns one `Vec<DataPoint>` per spec in the same order.
    pub fn multi_rollup(&self, points: &[DataPoint], specs: &[RollupSpec]) -> Vec<Vec<DataPoint>> {
        specs.iter().map(|spec| self.rollup(points, spec)).collect()
    }

    /// Downsample `points` to at most `target_count` points using the
    /// Largest-Triangle-Three-Buckets (LTTB) algorithm.
    ///
    /// Special cases:
    /// - `target_count == 0` → returns empty.
    /// - `target_count >= points.len()` → returns a clone of all points.
    /// - LTTB always preserves the first and last points.
    pub fn lttb(&self, points: &[DataPoint], target_count: usize) -> Vec<DataPoint> {
        let n = points.len();
        if target_count == 0 {
            return Vec::new();
        }
        if n <= target_count {
            return points.to_vec();
        }
        if target_count == 1 {
            return vec![points[0].clone()];
        }
        if target_count == 2 {
            return vec![points[0].clone(), points[n - 1].clone()];
        }

        // The bucket count excluding first and last points
        let bucket_count = target_count - 2;
        // Points to distribute across buckets (excluding first and last)
        let data = &points[1..n - 1];
        let data_len = data.len();

        let mut sampled: Vec<DataPoint> = Vec::with_capacity(target_count);
        sampled.push(points[0].clone());

        let mut last_selected = 0usize; // index in original `points`

        for b in 0..bucket_count {
            // Determine the range [a, b) for this bucket in `data`
            let a_idx = (b * data_len) / bucket_count;
            let b_idx = ((b + 1) * data_len) / bucket_count;
            let bucket_end = b_idx.min(data_len);
            let bucket_range = &data[a_idx..bucket_end];

            if bucket_range.is_empty() {
                continue;
            }

            // Calculate centroid of the NEXT bucket (or last point for last bucket)
            let next_a = ((b + 1) * data_len) / bucket_count;
            let next_b = ((b + 2) * data_len) / bucket_count;
            let avg_ts: f64;
            let avg_val: f64;

            if b + 1 >= bucket_count {
                // Use the last point as next "point"
                avg_ts = points[n - 1].timestamp_ms as f64;
                avg_val = points[n - 1].value;
            } else {
                let next_range = &data[next_a..next_b.min(data_len)];
                if next_range.is_empty() {
                    avg_ts = points[n - 1].timestamp_ms as f64;
                    avg_val = points[n - 1].value;
                } else {
                    avg_ts = next_range
                        .iter()
                        .map(|p| p.timestamp_ms as f64)
                        .sum::<f64>()
                        / next_range.len() as f64;
                    avg_val =
                        next_range.iter().map(|p| p.value).sum::<f64>() / next_range.len() as f64;
                }
            }

            // Current "last selected" point
            let last = &points[last_selected];
            let lx = last.timestamp_ms as f64;
            let ly = last.value;

            // Find the point in bucket_range that forms the largest triangle
            let mut best_area = -1.0f64;
            let mut best_local = 0usize;

            for (li, pt) in bucket_range.iter().enumerate() {
                let px = pt.timestamp_ms as f64;
                let py = pt.value;
                // Triangle area (absolute value of cross product / 2)
                let area = ((lx - avg_ts) * (py - ly) - (lx - px) * (avg_val - ly)).abs() * 0.5;
                if area > best_area {
                    best_area = area;
                    best_local = li;
                }
            }

            // Map best_local back to the index in `points`
            last_selected = 1 + a_idx + best_local; // +1 for skipped first point
            sampled.push(points[last_selected].clone());
        }

        sampled.push(points[n - 1].clone());
        sampled
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Apply `fn_` to a non-empty slice of values.
    fn aggregate(values: &[f64], fn_: RollupFn) -> f64 {
        match fn_ {
            RollupFn::Mean => {
                let sum: f64 = values.iter().sum();
                sum / values.len() as f64
            }
            RollupFn::Sum => values.iter().sum(),
            RollupFn::Min => values.iter().cloned().fold(f64::INFINITY, f64::min),
            RollupFn::Max => values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            RollupFn::Count => values.len() as f64,
            RollupFn::Last => *values.last().unwrap_or(&0.0),
            RollupFn::First => *values.first().unwrap_or(&0.0),
        }
    }
}

impl Default for RollupEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pts(data: &[(u64, f64)]) -> Vec<DataPoint> {
        data.iter().map(|(ts, v)| DataPoint::new(*ts, *v)).collect()
    }

    fn eng() -> RollupEngine {
        RollupEngine::new()
    }

    // ── RollupFn::Mean ────────────────────────────────────────────────────────

    #[test]
    fn test_rollup_mean_single_window() {
        let points = pts(&[(0, 2.0), (1, 4.0), (2, 6.0)]);
        let spec = RollupSpec::new(10, RollupFn::Mean);
        let result = eng().rollup(&points, &spec);
        assert_eq!(result.len(), 1);
        assert!((result[0].value - 4.0).abs() < 1e-9);
        assert_eq!(result[0].timestamp_ms, 0);
    }

    #[test]
    fn test_rollup_mean_two_windows() {
        let points = pts(&[(0, 2.0), (5, 4.0), (10, 6.0), (15, 8.0)]);
        let spec = RollupSpec::new(10, RollupFn::Mean);
        let result = eng().rollup(&points, &spec);
        assert_eq!(result.len(), 2);
        assert!((result[0].value - 3.0).abs() < 1e-9); // (2+4)/2
        assert!((result[1].value - 7.0).abs() < 1e-9); // (6+8)/2
    }

    // ── RollupFn::Sum ─────────────────────────────────────────────────────────

    #[test]
    fn test_rollup_sum() {
        let points = pts(&[(0, 1.0), (1, 2.0), (2, 3.0)]);
        let spec = RollupSpec::new(10, RollupFn::Sum);
        let result = eng().rollup(&points, &spec);
        assert_eq!(result.len(), 1);
        assert!((result[0].value - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_rollup_sum_two_windows() {
        let points = pts(&[(0, 1.0), (5, 2.0), (10, 3.0), (15, 4.0)]);
        let spec = RollupSpec::new(10, RollupFn::Sum);
        let result = eng().rollup(&points, &spec);
        assert_eq!(result.len(), 2);
        assert!((result[0].value - 3.0).abs() < 1e-9);
        assert!((result[1].value - 7.0).abs() < 1e-9);
    }

    // ── RollupFn::Min ─────────────────────────────────────────────────────────

    #[test]
    fn test_rollup_min() {
        let points = pts(&[(0, 5.0), (1, 2.0), (2, 8.0)]);
        let spec = RollupSpec::new(10, RollupFn::Min);
        let result = eng().rollup(&points, &spec);
        assert_eq!(result.len(), 1);
        assert!((result[0].value - 2.0).abs() < 1e-9);
    }

    // ── RollupFn::Max ─────────────────────────────────────────────────────────

    #[test]
    fn test_rollup_max() {
        let points = pts(&[(0, 5.0), (1, 2.0), (2, 8.0)]);
        let spec = RollupSpec::new(10, RollupFn::Max);
        let result = eng().rollup(&points, &spec);
        assert_eq!(result.len(), 1);
        assert!((result[0].value - 8.0).abs() < 1e-9);
    }

    // ── RollupFn::Count ───────────────────────────────────────────────────────

    #[test]
    fn test_rollup_count() {
        let points = pts(&[(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)]);
        let spec = RollupSpec::new(10, RollupFn::Count);
        let result = eng().rollup(&points, &spec);
        assert_eq!(result.len(), 1);
        assert!((result[0].value - 4.0).abs() < 1e-9);
    }

    // ── RollupFn::Last ────────────────────────────────────────────────────────

    #[test]
    fn test_rollup_last() {
        let points = pts(&[(0, 1.0), (1, 2.0), (2, 99.0)]);
        let spec = RollupSpec::new(10, RollupFn::Last);
        let result = eng().rollup(&points, &spec);
        assert_eq!(result.len(), 1);
        assert!((result[0].value - 99.0).abs() < 1e-9);
    }

    // ── RollupFn::First ───────────────────────────────────────────────────────

    #[test]
    fn test_rollup_first() {
        let points = pts(&[(0, 77.0), (1, 2.0), (2, 3.0)]);
        let spec = RollupSpec::new(10, RollupFn::First);
        let result = eng().rollup(&points, &spec);
        assert_eq!(result.len(), 1);
        assert!((result[0].value - 77.0).abs() < 1e-9);
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_rollup_empty_input() {
        let spec = RollupSpec::new(10, RollupFn::Mean);
        let result = eng().rollup(&[], &spec);
        assert!(result.is_empty());
    }

    #[test]
    fn test_rollup_single_point() {
        let points = pts(&[(5, 42.0)]);
        let spec = RollupSpec::new(10, RollupFn::Mean);
        let result = eng().rollup(&points, &spec);
        assert_eq!(result.len(), 1);
        assert!((result[0].value - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_rollup_window_timestamp_is_start_of_window() {
        let points = pts(&[(15, 1.0), (17, 2.0)]);
        let spec = RollupSpec::new(10, RollupFn::Mean);
        let result = eng().rollup(&points, &spec);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].timestamp_ms, 10); // floor(15/10)*10 = 10
    }

    #[test]
    fn test_rollup_multiple_windows_count() {
        let points = pts(&[(0, 1.0), (5, 2.0), (10, 3.0), (15, 4.0), (20, 5.0)]);
        let spec = RollupSpec::new(10, RollupFn::Count);
        let result = eng().rollup(&points, &spec);
        assert_eq!(result.len(), 3);
        assert!((result[0].value - 2.0).abs() < 1e-9);
        assert!((result[1].value - 2.0).abs() < 1e-9);
        assert!((result[2].value - 1.0).abs() < 1e-9);
    }

    // ── multi_rollup ──────────────────────────────────────────────────────────

    #[test]
    fn test_multi_rollup_empty_specs() {
        let points = pts(&[(0, 1.0)]);
        let result = eng().multi_rollup(&points, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_multi_rollup_two_specs() {
        let points = pts(&[(0, 1.0), (5, 2.0), (10, 3.0), (15, 4.0)]);
        let specs = vec![
            RollupSpec::new(10, RollupFn::Mean),
            RollupSpec::new(20, RollupFn::Sum),
        ];
        let results = eng().multi_rollup(&points, &specs);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2); // two 10ms windows
        assert_eq!(results[1].len(), 1); // one 20ms window
        assert!((results[1][0].value - 10.0).abs() < 1e-9); // 1+2+3+4=10
    }

    #[test]
    fn test_multi_rollup_different_aggregations() {
        let points = pts(&[(0, 2.0), (5, 4.0), (7, 6.0)]);
        let specs = vec![
            RollupSpec::new(10, RollupFn::Min),
            RollupSpec::new(10, RollupFn::Max),
            RollupSpec::new(10, RollupFn::Sum),
        ];
        let results = eng().multi_rollup(&points, &specs);
        assert_eq!(results.len(), 3);
        assert!((results[0][0].value - 2.0).abs() < 1e-9); // min
        assert!((results[1][0].value - 6.0).abs() < 1e-9); // max
        assert!((results[2][0].value - 12.0).abs() < 1e-9); // sum
    }

    // ── lttb ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_lttb_reduces_count() {
        let points = pts(&[
            (0, 0.0),
            (1, 1.0),
            (2, 0.5),
            (3, 2.0),
            (4, 1.5),
            (5, 3.0),
            (6, 2.5),
            (7, 4.0),
            (8, 3.5),
            (9, 5.0),
        ]);
        let result = eng().lttb(&points, 5);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_lttb_preserves_endpoints() {
        let points: Vec<DataPoint> = (0..20)
            .map(|i| DataPoint::new(i as u64, i as f64))
            .collect();
        let result = eng().lttb(&points, 7);
        assert_eq!(result.first().map(|p| p.timestamp_ms), Some(0));
        assert_eq!(result.last().map(|p| p.timestamp_ms), Some(19));
    }

    #[test]
    fn test_lttb_target_ge_len_returns_all() {
        let points: Vec<DataPoint> = (0..10)
            .map(|i| DataPoint::new(i as u64, i as f64))
            .collect();
        let result = eng().lttb(&points, 10);
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_lttb_target_greater_than_len() {
        let points: Vec<DataPoint> = (0..5).map(|i| DataPoint::new(i as u64, i as f64)).collect();
        let result = eng().lttb(&points, 100);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_lttb_target_zero_empty() {
        let points = pts(&[(0, 1.0), (1, 2.0)]);
        let result = eng().lttb(&points, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_lttb_target_two() {
        let points: Vec<DataPoint> = (0..10)
            .map(|i| DataPoint::new(i as u64, i as f64))
            .collect();
        let result = eng().lttb(&points, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].timestamp_ms, 0);
        assert_eq!(result[1].timestamp_ms, 9);
    }

    #[test]
    fn test_lttb_empty_input() {
        let result = eng().lttb(&[], 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_lttb_single_point() {
        let points = pts(&[(42, 1.0)]);
        let result = eng().lttb(&points, 5);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_lttb_two_points() {
        let points = pts(&[(0, 1.0), (10, 5.0)]);
        let result = eng().lttb(&points, 5);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_lttb_result_sorted_by_timestamp() {
        let points: Vec<DataPoint> = (0..50u64)
            .map(|i| DataPoint::new(i * 100, (i as f64).sin()))
            .collect();
        let result = eng().lttb(&points, 10);
        for w in result.windows(2) {
            assert!(w[0].timestamp_ms <= w[1].timestamp_ms);
        }
    }

    // ── DataPoint / RollupSpec helpers ────────────────────────────────────────

    #[test]
    fn test_data_point_new() {
        let dp = DataPoint::new(123, 4.56);
        assert_eq!(dp.timestamp_ms, 123);
        assert!((dp.value - 4.56).abs() < 1e-9);
    }

    #[test]
    fn test_rollup_spec_new() {
        let spec = RollupSpec::new(500, RollupFn::Sum);
        assert_eq!(spec.window_ms, 500);
        assert_eq!(spec.function, RollupFn::Sum);
    }

    #[test]
    fn test_rollup_engine_default() {
        let _ = RollupEngine;
    }
}
