//! Time-series downsampling algorithms.
//!
//! Provides multiple downsampling strategies for reducing the number of data points
//! in a time series while preserving meaningful characteristics:
//!
//! - **LTTB** (Largest-Triangle-Three-Buckets): Preserves visual shape
//! - **Average**: Mean value per bucket
//! - **MinMax**: Min and max per bucket (up to 2× target points)
//! - **First** / **Last**: First or last sample per bucket
//! - **Sum**: Sum of values per bucket
//! - **Count**: Number of samples per bucket

/// A single time-series data point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DataPoint {
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
    /// Observed value.
    pub value: f64,
}

impl DataPoint {
    /// Create a new data point.
    pub fn new(timestamp: i64, value: f64) -> Self {
        DataPoint { timestamp, value }
    }
}

/// Downsampling method to apply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DownsampleMethod {
    /// Largest-Triangle-Three-Buckets: best visual fidelity.
    Lttb,
    /// Arithmetic mean per bucket.
    Average,
    /// Min and max per bucket (returns up to 2 × target points).
    MinMax,
    /// First point per bucket.
    First,
    /// Last point per bucket.
    Last,
    /// Sum of all values per bucket.
    Sum,
    /// Count of samples in each bucket, stored as `f64`.
    Count,
}

/// Configuration for a downsampling operation.
#[derive(Debug, Clone)]
pub struct DownsampleConfig {
    /// Desired maximum number of output points.
    pub target_points: usize,
    /// Algorithm to use.
    pub method: DownsampleMethod,
}

impl DownsampleConfig {
    pub fn new(target_points: usize, method: DownsampleMethod) -> Self {
        DownsampleConfig {
            target_points,
            method,
        }
    }
}

/// Stateless downsampling engine.
pub struct Downsampler;

impl Downsampler {
    /// Dispatch to the correct algorithm based on `config.method`.
    pub fn downsample(data: &[DataPoint], config: &DownsampleConfig) -> Vec<DataPoint> {
        let target = config.target_points;
        match config.method {
            DownsampleMethod::Lttb => Self::lttb(data, target),
            DownsampleMethod::Average => Self::bucket_average(data, target),
            DownsampleMethod::MinMax => Self::bucket_min_max(data, target),
            DownsampleMethod::First => Self::bucket_first(data, target),
            DownsampleMethod::Last => Self::bucket_last(data, target),
            DownsampleMethod::Sum => Self::bucket_sum(data, target),
            DownsampleMethod::Count => Self::bucket_count(data, target),
        }
    }

    // ── LTTB ─────────────────────────────────────────────────────────────────

    /// Largest-Triangle-Three-Buckets downsampling.
    ///
    /// Always keeps the first and last point. Divides the remaining data into
    /// `target - 2` buckets and selects the point in each bucket that forms the
    /// largest triangle with the selected point from the previous bucket and the
    /// centroid of the next bucket.
    pub fn lttb(data: &[DataPoint], target: usize) -> Vec<DataPoint> {
        if target == 0 {
            return vec![];
        }
        if data.len() <= target {
            return data.to_vec();
        }
        if target <= 2 {
            // Return first and last (or just first if target == 1)
            if target == 1 {
                return vec![data[0]];
            }
            return vec![data[0], *data.last().unwrap()];
        }

        let n = data.len();
        let mut result = Vec::with_capacity(target);

        // Always include first point
        result.push(data[0]);

        // Number of "middle" buckets
        let bucket_count = target - 2;
        let bucket_size = (n - 2) as f64 / bucket_count as f64;

        let mut last_selected = 0usize; // index of the last selected point

        for i in 0..bucket_count {
            // Current bucket range
            let bucket_start = (1.0 + i as f64 * bucket_size) as usize;
            let bucket_end = ((1.0 + (i + 1) as f64 * bucket_size) as usize).min(n - 1);

            // Next bucket centroid (for look-ahead)
            let next_start = bucket_end;
            let next_end = if i + 1 < bucket_count {
                ((1.0 + (i + 2) as f64 * bucket_size) as usize).min(n - 1)
            } else {
                n - 1
            };

            let centroid_x: f64 = data[next_start..=next_end]
                .iter()
                .map(|p| p.timestamp as f64)
                .sum::<f64>()
                / (next_end - next_start + 1) as f64;
            let centroid_y: f64 = data[next_start..=next_end]
                .iter()
                .map(|p| p.value)
                .sum::<f64>()
                / (next_end - next_start + 1) as f64;

            let a = &data[last_selected];

            // Find the point in [bucket_start, bucket_end) with max triangle area
            let mut max_area = -1.0_f64;
            let mut selected = bucket_start;
            let end_exclusive = bucket_end.max(bucket_start + 1);
            for (j, p) in data
                .iter()
                .enumerate()
                .skip(bucket_start)
                .take(end_exclusive - bucket_start)
            {
                let area = Self::triangle_area(
                    a.timestamp as f64,
                    a.value,
                    p.timestamp as f64,
                    p.value,
                    centroid_x,
                    centroid_y,
                );
                if area > max_area {
                    max_area = area;
                    selected = j;
                }
            }

            result.push(data[selected]);
            last_selected = selected;
        }

        // Always include last point
        result.push(*data.last().unwrap());

        result
    }

    // ── Bucket helpers ────────────────────────────────────────────────────────

    /// Divide data into approximately `target` equal-width buckets.
    /// Returns (start_inclusive, end_exclusive) slices.
    fn buckets(n: usize, target: usize) -> Vec<(usize, usize)> {
        if n == 0 || target == 0 {
            return vec![];
        }
        let bucket_count = target.min(n);
        let mut result = Vec::with_capacity(bucket_count);
        for i in 0..bucket_count {
            let start = (i * n) / bucket_count;
            let end = ((i + 1) * n) / bucket_count;
            if start < end {
                result.push((start, end));
            }
        }
        result
    }

    /// Arithmetic mean per bucket. Output has at most `target` points.
    pub fn bucket_average(data: &[DataPoint], target: usize) -> Vec<DataPoint> {
        if target == 0 {
            return vec![];
        }
        if data.is_empty() || data.len() <= target {
            return data.to_vec();
        }
        Self::buckets(data.len(), target)
            .into_iter()
            .map(|(start, end)| {
                let bucket = &data[start..end];
                let sum: f64 = bucket.iter().map(|p| p.value).sum();
                let avg = sum / bucket.len() as f64;
                let mid_ts = bucket[bucket.len() / 2].timestamp;
                DataPoint::new(mid_ts, avg)
            })
            .collect()
    }

    /// Min and max per bucket. Returns at most `2 × target` points.
    pub fn bucket_min_max(data: &[DataPoint], target: usize) -> Vec<DataPoint> {
        if target == 0 {
            return vec![];
        }
        if data.is_empty() || data.len() <= target {
            return data.to_vec();
        }
        let mut result = Vec::new();
        for (start, end) in Self::buckets(data.len(), target) {
            let bucket = &data[start..end];
            let min = bucket.iter().copied().min_by(|a, b| {
                a.value
                    .partial_cmp(&b.value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let max = bucket.iter().copied().max_by(|a, b| {
                a.value
                    .partial_cmp(&b.value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if let Some(mn) = min {
                if let Some(mx) = max {
                    if mn.timestamp <= mx.timestamp {
                        result.push(mn);
                        if mn.timestamp != mx.timestamp {
                            result.push(mx);
                        }
                    } else {
                        result.push(mx);
                        if mn.timestamp != mx.timestamp {
                            result.push(mn);
                        }
                    }
                }
            }
        }
        result
    }

    /// First sample per bucket.
    pub fn bucket_first(data: &[DataPoint], target: usize) -> Vec<DataPoint> {
        if target == 0 {
            return vec![];
        }
        if data.is_empty() || data.len() <= target {
            return data.to_vec();
        }
        Self::buckets(data.len(), target)
            .into_iter()
            .map(|(start, _end)| data[start])
            .collect()
    }

    /// Last sample per bucket.
    pub fn bucket_last(data: &[DataPoint], target: usize) -> Vec<DataPoint> {
        if target == 0 {
            return vec![];
        }
        if data.is_empty() || data.len() <= target {
            return data.to_vec();
        }
        Self::buckets(data.len(), target)
            .into_iter()
            .map(|(_start, end)| data[end - 1])
            .collect()
    }

    /// Sum of values per bucket.
    pub fn bucket_sum(data: &[DataPoint], target: usize) -> Vec<DataPoint> {
        if target == 0 {
            return vec![];
        }
        if data.is_empty() || data.len() <= target {
            return data.to_vec();
        }
        Self::buckets(data.len(), target)
            .into_iter()
            .map(|(start, end)| {
                let bucket = &data[start..end];
                let sum: f64 = bucket.iter().map(|p| p.value).sum();
                let mid_ts = bucket[bucket.len() / 2].timestamp;
                DataPoint::new(mid_ts, sum)
            })
            .collect()
    }

    /// Count of samples per bucket (value = count as f64).
    pub fn bucket_count(data: &[DataPoint], target: usize) -> Vec<DataPoint> {
        if target == 0 {
            return vec![];
        }
        if data.is_empty() || data.len() <= target {
            return data.to_vec();
        }
        Self::buckets(data.len(), target)
            .into_iter()
            .map(|(start, end)| {
                let bucket = &data[start..end];
                let count = bucket.len() as f64;
                let mid_ts = bucket[bucket.len() / 2].timestamp;
                DataPoint::new(mid_ts, count)
            })
            .collect()
    }

    // ── Geometry ──────────────────────────────────────────────────────────────

    /// Absolute area of the triangle formed by three points.
    pub fn triangle_area(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> f64 {
        ((ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) / 2.0).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_points(pairs: &[(i64, f64)]) -> Vec<DataPoint> {
        pairs.iter().map(|&(t, v)| DataPoint::new(t, v)).collect()
    }

    fn linear_data(n: usize) -> Vec<DataPoint> {
        (0..n).map(|i| DataPoint::new(i as i64, i as f64)).collect()
    }

    fn timestamps_monotone(points: &[DataPoint]) -> bool {
        points.windows(2).all(|w| w[0].timestamp <= w[1].timestamp)
    }

    // ── Edge cases: empty / single / equal / target=0 / target≥len ────────────

    #[test]
    fn test_lttb_empty() {
        assert!(Downsampler::lttb(&[], 10).is_empty());
    }

    #[test]
    fn test_lttb_target_zero() {
        let data = linear_data(10);
        assert!(Downsampler::lttb(&data, 0).is_empty());
    }

    #[test]
    fn test_lttb_target_ge_len() {
        let data = linear_data(5);
        let result = Downsampler::lttb(&data, 10);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_lttb_single_point() {
        let data = vec![DataPoint::new(1, 1.0)];
        let result = Downsampler::lttb(&data, 5);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_lttb_keeps_first_and_last() {
        let data = linear_data(100);
        let result = Downsampler::lttb(&data, 10);
        assert_eq!(result[0].timestamp, 0);
        assert_eq!(result.last().unwrap().timestamp, 99);
    }

    #[test]
    fn test_lttb_output_count() {
        let data = linear_data(1000);
        let result = Downsampler::lttb(&data, 50);
        assert_eq!(result.len(), 50);
    }

    #[test]
    fn test_lttb_monotone_timestamps() {
        let data = linear_data(200);
        let result = Downsampler::lttb(&data, 30);
        assert!(timestamps_monotone(&result));
    }

    #[test]
    fn test_lttb_target_two() {
        let data = linear_data(10);
        let result = Downsampler::lttb(&data, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].timestamp, 0);
        assert_eq!(result[1].timestamp, 9);
    }

    #[test]
    fn test_lttb_target_one() {
        let data = linear_data(10);
        let result = Downsampler::lttb(&data, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].timestamp, 0);
    }

    // ── Average ───────────────────────────────────────────────────────────────

    #[test]
    fn test_average_empty() {
        assert!(Downsampler::bucket_average(&[], 5).is_empty());
    }

    #[test]
    fn test_average_target_zero() {
        let data = linear_data(10);
        assert!(Downsampler::bucket_average(&data, 0).is_empty());
    }

    #[test]
    fn test_average_target_ge_len() {
        let data = linear_data(5);
        let result = Downsampler::bucket_average(&data, 10);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_average_output_count() {
        let data = linear_data(100);
        let result = Downsampler::bucket_average(&data, 10);
        assert!(result.len() <= 10);
    }

    #[test]
    fn test_average_two_equal_buckets() {
        let data = make_points(&[(0, 1.0), (1, 3.0), (2, 5.0), (3, 7.0)]);
        let result = Downsampler::bucket_average(&data, 2);
        assert_eq!(result.len(), 2);
        assert!((result[0].value - 2.0).abs() < 1e-9); // (1+3)/2
        assert!((result[1].value - 6.0).abs() < 1e-9); // (5+7)/2
    }

    #[test]
    fn test_average_monotone_timestamps() {
        let data = linear_data(100);
        let result = Downsampler::bucket_average(&data, 20);
        assert!(timestamps_monotone(&result));
    }

    // ── MinMax ────────────────────────────────────────────────────────────────

    #[test]
    fn test_min_max_empty() {
        assert!(Downsampler::bucket_min_max(&[], 5).is_empty());
    }

    #[test]
    fn test_min_max_target_zero() {
        let data = linear_data(10);
        assert!(Downsampler::bucket_min_max(&data, 0).is_empty());
    }

    #[test]
    fn test_min_max_at_most_two_times_target() {
        let data = linear_data(100);
        let result = Downsampler::bucket_min_max(&data, 10);
        assert!(result.len() <= 20);
    }

    #[test]
    fn test_min_max_single_point_per_bucket() {
        let data = make_points(&[(0, 5.0), (1, 3.0), (2, 7.0), (3, 1.0)]);
        // 2 buckets → at most 4 points
        let result = Downsampler::bucket_min_max(&data, 2);
        assert!(result.len() <= 4);
    }

    #[test]
    fn test_min_max_target_ge_len() {
        let data = linear_data(5);
        let result = Downsampler::bucket_min_max(&data, 10);
        assert_eq!(result.len(), 5);
    }

    // ── First ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_first_empty() {
        assert!(Downsampler::bucket_first(&[], 5).is_empty());
    }

    #[test]
    fn test_first_target_zero() {
        let data = linear_data(10);
        assert!(Downsampler::bucket_first(&data, 0).is_empty());
    }

    #[test]
    fn test_first_output_count() {
        let data = linear_data(100);
        let result = Downsampler::bucket_first(&data, 10);
        assert!(result.len() <= 10);
    }

    #[test]
    fn test_first_returns_first_of_bucket() {
        let data = make_points(&[(0, 10.0), (1, 20.0), (2, 30.0), (3, 40.0)]);
        let result = Downsampler::bucket_first(&data, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].value, 10.0);
        assert_eq!(result[1].value, 30.0);
    }

    #[test]
    fn test_first_target_ge_len() {
        let data = linear_data(5);
        let result = Downsampler::bucket_first(&data, 20);
        assert_eq!(result.len(), 5);
    }

    // ── Last ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_last_empty() {
        assert!(Downsampler::bucket_last(&[], 5).is_empty());
    }

    #[test]
    fn test_last_target_zero() {
        let data = linear_data(10);
        assert!(Downsampler::bucket_last(&data, 0).is_empty());
    }

    #[test]
    fn test_last_returns_last_of_bucket() {
        let data = make_points(&[(0, 10.0), (1, 20.0), (2, 30.0), (3, 40.0)]);
        let result = Downsampler::bucket_last(&data, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].value, 20.0);
        assert_eq!(result[1].value, 40.0);
    }

    #[test]
    fn test_last_target_ge_len() {
        let data = linear_data(5);
        let result = Downsampler::bucket_last(&data, 20);
        assert_eq!(result.len(), 5);
    }

    // ── Sum ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_sum_empty() {
        assert!(Downsampler::bucket_sum(&[], 5).is_empty());
    }

    #[test]
    fn test_sum_target_zero() {
        let data = linear_data(10);
        assert!(Downsampler::bucket_sum(&data, 0).is_empty());
    }

    #[test]
    fn test_sum_correct_values() {
        let data = make_points(&[(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)]);
        let result = Downsampler::bucket_sum(&data, 2);
        assert_eq!(result.len(), 2);
        assert!((result[0].value - 3.0).abs() < 1e-9); // 1+2
        assert!((result[1].value - 7.0).abs() < 1e-9); // 3+4
    }

    #[test]
    fn test_sum_target_ge_len() {
        let data = linear_data(5);
        let result = Downsampler::bucket_sum(&data, 20);
        assert_eq!(result.len(), 5);
    }

    // ── Count ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_count_empty() {
        assert!(Downsampler::bucket_count(&[], 5).is_empty());
    }

    #[test]
    fn test_count_target_zero() {
        let data = linear_data(10);
        assert!(Downsampler::bucket_count(&data, 0).is_empty());
    }

    #[test]
    fn test_count_correct_counts() {
        let data = make_points(&[(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)]);
        let result = Downsampler::bucket_count(&data, 2);
        assert_eq!(result.len(), 2);
        assert!((result[0].value - 2.0).abs() < 1e-9);
        assert!((result[1].value - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_count_total_preserved() {
        let data = linear_data(60);
        let result = Downsampler::bucket_count(&data, 6);
        let total: f64 = result.iter().map(|p| p.value).sum();
        assert!((total - 60.0).abs() < 1e-9);
    }

    #[test]
    fn test_count_target_ge_len() {
        let data = linear_data(5);
        let result = Downsampler::bucket_count(&data, 20);
        assert_eq!(result.len(), 5);
    }

    // ── downsample dispatch ───────────────────────────────────────────────────

    #[test]
    fn test_downsample_dispatch_lttb() {
        let data = linear_data(100);
        let config = DownsampleConfig::new(10, DownsampleMethod::Lttb);
        let result = Downsampler::downsample(&data, &config);
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_downsample_dispatch_average() {
        let data = linear_data(100);
        let config = DownsampleConfig::new(10, DownsampleMethod::Average);
        let result = Downsampler::downsample(&data, &config);
        assert!(result.len() <= 10);
    }

    #[test]
    fn test_downsample_dispatch_min_max() {
        let data = linear_data(100);
        let config = DownsampleConfig::new(10, DownsampleMethod::MinMax);
        let result = Downsampler::downsample(&data, &config);
        assert!(result.len() <= 20);
    }

    #[test]
    fn test_downsample_dispatch_first() {
        let data = linear_data(100);
        let config = DownsampleConfig::new(10, DownsampleMethod::First);
        let result = Downsampler::downsample(&data, &config);
        assert!(result.len() <= 10);
    }

    #[test]
    fn test_downsample_dispatch_last() {
        let data = linear_data(100);
        let config = DownsampleConfig::new(10, DownsampleMethod::Last);
        let result = Downsampler::downsample(&data, &config);
        assert!(result.len() <= 10);
    }

    #[test]
    fn test_downsample_dispatch_sum() {
        let data = linear_data(100);
        let config = DownsampleConfig::new(10, DownsampleMethod::Sum);
        let result = Downsampler::downsample(&data, &config);
        assert!(result.len() <= 10);
    }

    #[test]
    fn test_downsample_dispatch_count() {
        let data = linear_data(100);
        let config = DownsampleConfig::new(10, DownsampleMethod::Count);
        let result = Downsampler::downsample(&data, &config);
        assert!(result.len() <= 10);
    }

    // ── triangle_area ─────────────────────────────────────────────────────────

    #[test]
    fn test_triangle_area_right_triangle() {
        // (0,0), (1,0), (0,1) → area = 0.5
        let area = Downsampler::triangle_area(0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
        assert!((area - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_triangle_area_collinear_zero() {
        // Three collinear points have zero area
        let area = Downsampler::triangle_area(0.0, 0.0, 1.0, 1.0, 2.0, 2.0);
        assert!(area < 1e-9);
    }

    #[test]
    fn test_triangle_area_symmetric() {
        let a = Downsampler::triangle_area(0.0, 0.0, 1.0, 2.0, 3.0, 0.0);
        let b = Downsampler::triangle_area(3.0, 0.0, 1.0, 2.0, 0.0, 0.0);
        assert!((a - b).abs() < 1e-9);
    }
}
