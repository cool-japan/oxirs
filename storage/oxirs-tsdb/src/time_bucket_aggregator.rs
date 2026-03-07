//! Fixed-interval and calendar-based time bucket aggregation.
//!
//! [`TimeBucketAggregator`] groups [`DataPoint`]s into time buckets of a
//! specified [`BucketInterval`] and computes per-bucket statistics
//! (count, sum, min, max, first, last).

// ── DataPoint ────────────────────────────────────────────────────────────────

/// A single timestamped measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct DataPoint {
    /// Unix timestamp in **milliseconds**.
    pub timestamp: i64,
    /// The measured value.
    pub value: f64,
}

impl DataPoint {
    /// Convenience constructor.
    pub fn new(timestamp: i64, value: f64) -> Self {
        Self { timestamp, value }
    }
}

// ── Bucket ───────────────────────────────────────────────────────────────────

/// Aggregated statistics for a single time bucket.
#[derive(Clone, Debug, PartialEq)]
pub struct Bucket {
    /// Start of the bucket interval (inclusive), in Unix milliseconds.
    pub start: i64,
    /// End of the bucket interval (exclusive), in Unix milliseconds.
    pub end: i64,
    /// Number of data points that fell into this bucket.
    pub count: usize,
    /// Sum of all values.
    pub sum: f64,
    /// Minimum value (or `f64::INFINITY` if empty).
    pub min: f64,
    /// Maximum value (or `f64::NEG_INFINITY` if empty).
    pub max: f64,
    /// Value of the first (earliest) data point in the bucket.
    pub first: f64,
    /// Value of the last (latest) data point in the bucket.
    pub last: f64,
}

impl Bucket {
    /// Arithmetic mean, or `0.0` if the bucket is empty.
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    /// Returns `true` if no data points were placed in this bucket.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    fn new_empty(start: i64, end: i64) -> Self {
        Self {
            start,
            end,
            count: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            first: 0.0,
            last: 0.0,
        }
    }

    fn push(&mut self, value: f64) {
        if self.count == 0 {
            self.first = value;
            self.min = value;
            self.max = value;
        } else {
            if value < self.min {
                self.min = value;
            }
            if value > self.max {
                self.max = value;
            }
        }
        self.sum += value;
        self.last = value;
        self.count += 1;
    }
}

// ── BucketInterval ───────────────────────────────────────────────────────────

/// Specifies the width of a time bucket.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BucketInterval {
    /// Fixed duration in milliseconds.
    Fixed(i64),
    /// N minutes expressed as fixed milliseconds.
    Minutes(i64),
    /// N hours expressed as fixed milliseconds.
    Hours(i64),
    /// N days expressed as fixed milliseconds.
    Days(i64),
    /// N weeks expressed as fixed milliseconds.
    Weeks(i64),
}

impl BucketInterval {
    /// Convert the interval to a number of milliseconds.
    pub fn to_ms(&self) -> i64 {
        match self {
            BucketInterval::Fixed(ms) => *ms,
            BucketInterval::Minutes(n) => n * 60_000,
            BucketInterval::Hours(n) => n * 3_600_000,
            BucketInterval::Days(n) => n * 86_400_000,
            BucketInterval::Weeks(n) => n * 7 * 86_400_000,
        }
    }
}

// ── AggregationResult ────────────────────────────────────────────────────────

/// The output of an aggregation run.
#[derive(Clone, Debug)]
pub struct AggregationResult {
    /// The computed buckets, in chronological order.
    pub buckets: Vec<Bucket>,
    /// Total number of data points that were aggregated.
    pub total_points: usize,
    /// Number of buckets that received no data points.
    pub empty_buckets: usize,
}

// ── TimeBucketAggregator ─────────────────────────────────────────────────────

/// Stateless aggregation helpers.
pub struct TimeBucketAggregator;

impl TimeBucketAggregator {
    /// Aggregate `data` into fixed-width buckets over `[start, end)`.
    ///
    /// Data points outside `[start, end)` are silently ignored.
    pub fn aggregate(
        data: &[DataPoint],
        interval: &BucketInterval,
        start: i64,
        end: i64,
    ) -> AggregationResult {
        let interval_ms = interval.to_ms().max(1);

        // Build bucket vector.
        let n_buckets = ((end - start) as f64 / interval_ms as f64).ceil() as usize;
        let n_buckets = n_buckets.max(1);
        let mut buckets: Vec<Bucket> = (0..n_buckets)
            .map(|i| {
                let b_start = start + (i as i64) * interval_ms;
                let b_end = b_start + interval_ms;
                Bucket::new_empty(b_start, b_end.min(end))
            })
            .collect();

        let mut total_points = 0usize;

        for dp in data {
            if dp.timestamp < start || dp.timestamp >= end {
                continue;
            }
            let idx = ((dp.timestamp - start) / interval_ms) as usize;
            if idx < buckets.len() {
                buckets[idx].push(dp.value);
                total_points += 1;
            }
        }

        let empty_buckets = buckets.iter().filter(|b| b.is_empty()).count();

        AggregationResult {
            buckets,
            total_points,
            empty_buckets,
        }
    }

    /// Automatically compute an interval that produces approximately
    /// `target_buckets` buckets and then aggregate the data.
    ///
    /// If `data` is empty an empty result with a single empty bucket is
    /// returned.
    pub fn aggregate_auto(data: &[DataPoint], target_buckets: usize) -> AggregationResult {
        let target_buckets = target_buckets.max(1);

        if data.is_empty() {
            let empty_bucket = Bucket::new_empty(0, 1);
            return AggregationResult {
                buckets: vec![empty_bucket],
                total_points: 0,
                empty_buckets: 1,
            };
        }

        let min_ts = data.iter().map(|dp| dp.timestamp).min().unwrap_or(0);
        let max_ts = data.iter().map(|dp| dp.timestamp).max().unwrap_or(1);
        let range = (max_ts - min_ts).max(1);
        let interval_ms = (range as f64 / target_buckets as f64).ceil() as i64;
        let interval_ms = interval_ms.max(1);

        // Expand end by one interval to include the max point.
        let end = max_ts + interval_ms;
        Self::aggregate(data, &BucketInterval::Fixed(interval_ms), min_ts, end)
    }

    /// Compute the bucket-aligned start for a given timestamp and interval.
    pub fn bucket_start(timestamp: i64, interval_ms: i64) -> i64 {
        let interval_ms = interval_ms.max(1);
        if timestamp >= 0 {
            (timestamp / interval_ms) * interval_ms
        } else {
            // Correct floor division for negative timestamps.
            ((timestamp - interval_ms + 1) / interval_ms) * interval_ms
        }
    }

    /// Fill empty buckets in `result` with a synthetic point at `fill_value`.
    ///
    /// All statistics (sum, min, max, first, last) of filled buckets are set
    /// to `fill_value` and the count is set to 1.
    pub fn fill_gaps(result: &AggregationResult, fill_value: f64) -> AggregationResult {
        let buckets: Vec<Bucket> = result
            .buckets
            .iter()
            .map(|b| {
                if b.is_empty() {
                    let mut filled = b.clone();
                    filled.count = 1;
                    filled.sum = fill_value;
                    filled.min = fill_value;
                    filled.max = fill_value;
                    filled.first = fill_value;
                    filled.last = fill_value;
                    filled
                } else {
                    b.clone()
                }
            })
            .collect();

        let empty_buckets = buckets.iter().filter(|b| b.count == 0).count();
        let total_points = buckets.iter().map(|b| b.count).sum();

        AggregationResult {
            buckets,
            total_points,
            empty_buckets,
        }
    }

    /// Convert each bucket to a single [`DataPoint`] at its start time with
    /// the bucket mean as the value.
    ///
    /// Empty buckets (count == 0) produce a point with value `0.0`.
    pub fn downsample_buckets(result: &AggregationResult) -> Vec<DataPoint> {
        result
            .buckets
            .iter()
            .map(|b| DataPoint::new(b.start, b.mean()))
            .collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pts(pairs: &[(i64, f64)]) -> Vec<DataPoint> {
        pairs.iter().map(|&(t, v)| DataPoint::new(t, v)).collect()
    }

    // ── Bucket helpers ───────────────────────────────────────────────────────

    #[test]
    fn test_empty_bucket_mean_is_zero() {
        let b = Bucket::new_empty(0, 1000);
        assert_eq!(b.mean(), 0.0);
    }

    #[test]
    fn test_bucket_is_empty() {
        let b = Bucket::new_empty(0, 1000);
        assert!(b.is_empty());
    }

    #[test]
    fn test_bucket_push_updates_stats() {
        let mut b = Bucket::new_empty(0, 1000);
        b.push(3.0);
        b.push(7.0);
        b.push(5.0);
        assert_eq!(b.count, 3);
        assert_eq!(b.sum, 15.0);
        assert_eq!(b.min, 3.0);
        assert_eq!(b.max, 7.0);
        assert_eq!(b.first, 3.0);
        assert_eq!(b.last, 5.0);
    }

    #[test]
    fn test_bucket_mean_non_empty() {
        let mut b = Bucket::new_empty(0, 1000);
        b.push(2.0);
        b.push(4.0);
        assert!((b.mean() - 3.0).abs() < f64::EPSILON);
    }

    // ── BucketInterval::to_ms ────────────────────────────────────────────────

    #[test]
    fn test_fixed_interval_ms() {
        assert_eq!(BucketInterval::Fixed(500).to_ms(), 500);
    }

    #[test]
    fn test_minutes_interval() {
        assert_eq!(BucketInterval::Minutes(5).to_ms(), 300_000);
    }

    #[test]
    fn test_hours_interval() {
        assert_eq!(BucketInterval::Hours(2).to_ms(), 7_200_000);
    }

    #[test]
    fn test_days_interval() {
        assert_eq!(BucketInterval::Days(1).to_ms(), 86_400_000);
    }

    #[test]
    fn test_weeks_interval() {
        assert_eq!(BucketInterval::Weeks(1).to_ms(), 7 * 86_400_000);
    }

    // ── aggregate: basic ─────────────────────────────────────────────────────

    #[test]
    fn test_aggregate_single_point() {
        let data = pts(&[(500, 1.0)]);
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 2000);
        assert_eq!(res.total_points, 1);
        assert_eq!(res.buckets[0].count, 1);
        assert_eq!(res.buckets[0].sum, 1.0);
    }

    #[test]
    fn test_aggregate_multiple_points_same_bucket() {
        let data = pts(&[(100, 1.0), (200, 2.0), (300, 3.0)]);
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 1000);
        assert_eq!(res.buckets[0].count, 3);
        assert_eq!(res.buckets[0].sum, 6.0);
        assert_eq!(res.buckets[0].min, 1.0);
        assert_eq!(res.buckets[0].max, 3.0);
    }

    #[test]
    fn test_aggregate_multiple_buckets() {
        let data = pts(&[(100, 1.0), (1100, 2.0), (2100, 3.0)]);
        let res =
            TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 3000);
        assert_eq!(res.buckets.len(), 3);
        assert_eq!(res.buckets[0].count, 1);
        assert_eq!(res.buckets[1].count, 1);
        assert_eq!(res.buckets[2].count, 1);
    }

    #[test]
    fn test_aggregate_out_of_range_ignored() {
        let data = pts(&[(-1, 99.0), (5000, 99.0), (500, 1.0)]);
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 1000);
        assert_eq!(res.total_points, 1);
        assert_eq!(res.buckets[0].sum, 1.0);
    }

    #[test]
    fn test_aggregate_empty_data() {
        let data: Vec<DataPoint> = vec![];
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 3000);
        assert_eq!(res.total_points, 0);
        assert_eq!(res.empty_buckets, res.buckets.len());
    }

    #[test]
    fn test_aggregate_empty_buckets_counted() {
        let data = pts(&[(100, 1.0)]);
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 5000);
        assert_eq!(res.empty_buckets, 4); // 4 of 5 buckets are empty
    }

    #[test]
    fn test_aggregate_minutes_interval() {
        // 5-minute buckets over 15 minutes.
        let data = pts(&[
            (0, 1.0),
            (60_000, 2.0),       // minute 1 → bucket 0
            (300_001, 3.0),      // minute 5 → bucket 1
            (600_001, 4.0),      // minute 10 → bucket 2
        ]);
        let end = 15 * 60_000;
        let res =
            TimeBucketAggregator::aggregate(&data, &BucketInterval::Minutes(5), 0, end);
        assert_eq!(res.buckets.len(), 3);
        assert_eq!(res.buckets[0].count, 2);
        assert_eq!(res.buckets[1].count, 1);
        assert_eq!(res.buckets[2].count, 1);
    }

    #[test]
    fn test_aggregate_hours_interval() {
        let data = pts(&[
            (0, 10.0),
            (3_600_000, 20.0), // exactly 1 hour = start of bucket 1
        ]);
        let res =
            TimeBucketAggregator::aggregate(&data, &BucketInterval::Hours(1), 0, 7_200_000);
        assert_eq!(res.buckets[0].count, 1);
        assert_eq!(res.buckets[1].count, 1);
    }

    #[test]
    fn test_aggregate_days_interval() {
        let one_day = 86_400_000i64;
        let data = pts(&[(0, 5.0), (one_day, 10.0)]);
        let res =
            TimeBucketAggregator::aggregate(&data, &BucketInterval::Days(1), 0, 2 * one_day);
        assert_eq!(res.buckets.len(), 2);
        assert_eq!(res.buckets[0].sum, 5.0);
        assert_eq!(res.buckets[1].sum, 10.0);
    }

    #[test]
    fn test_aggregate_first_and_last() {
        let data = pts(&[(100, 10.0), (200, 20.0), (300, 30.0)]);
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 1000);
        assert_eq!(res.buckets[0].first, 10.0);
        assert_eq!(res.buckets[0].last, 30.0);
    }

    // ── aggregate_auto ───────────────────────────────────────────────────────

    #[test]
    fn test_aggregate_auto_empty() {
        let res = TimeBucketAggregator::aggregate_auto(&[], 10);
        assert_eq!(res.total_points, 0);
        assert_eq!(res.buckets.len(), 1);
        assert_eq!(res.empty_buckets, 1);
    }

    #[test]
    fn test_aggregate_auto_single_point() {
        let data = pts(&[(1000, 42.0)]);
        let res = TimeBucketAggregator::aggregate_auto(&data, 5);
        assert_eq!(res.total_points, 1);
    }

    #[test]
    fn test_aggregate_auto_produces_target_buckets_approx() {
        let data: Vec<DataPoint> = (0..100).map(|i| DataPoint::new(i * 10, i as f64)).collect();
        let res = TimeBucketAggregator::aggregate_auto(&data, 10);
        // Should produce roughly 10 buckets (may be slightly more).
        assert!(res.buckets.len() >= 9 && res.buckets.len() <= 12);
    }

    // ── bucket_start ─────────────────────────────────────────────────────────

    #[test]
    fn test_bucket_start_aligned() {
        assert_eq!(TimeBucketAggregator::bucket_start(0, 1000), 0);
        assert_eq!(TimeBucketAggregator::bucket_start(1000, 1000), 1000);
        assert_eq!(TimeBucketAggregator::bucket_start(2000, 1000), 2000);
    }

    #[test]
    fn test_bucket_start_mid_bucket() {
        assert_eq!(TimeBucketAggregator::bucket_start(500, 1000), 0);
        assert_eq!(TimeBucketAggregator::bucket_start(1500, 1000), 1000);
    }

    #[test]
    fn test_bucket_start_negative() {
        // -500 with 1000ms interval should start at -1000.
        assert_eq!(TimeBucketAggregator::bucket_start(-500, 1000), -1000);
        assert_eq!(TimeBucketAggregator::bucket_start(-1000, 1000), -1000);
    }

    // ── fill_gaps ────────────────────────────────────────────────────────────

    #[test]
    fn test_fill_gaps_fills_empty_buckets() {
        let data = pts(&[(0, 1.0)]);
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 3000);
        let filled = TimeBucketAggregator::fill_gaps(&res, 0.0);
        assert_eq!(filled.empty_buckets, 0);
        assert_eq!(filled.buckets[1].mean(), 0.0);
        assert_eq!(filled.buckets[2].mean(), 0.0);
    }

    #[test]
    fn test_fill_gaps_does_not_alter_non_empty() {
        let data = pts(&[(500, 7.0)]);
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 2000);
        let filled = TimeBucketAggregator::fill_gaps(&res, 99.0);
        assert_eq!(filled.buckets[0].sum, 7.0);
    }

    #[test]
    fn test_fill_gaps_custom_fill_value() {
        let data: Vec<DataPoint> = vec![];
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 2000);
        let filled = TimeBucketAggregator::fill_gaps(&res, -1.0);
        assert!(filled.buckets.iter().all(|b| b.sum == -1.0));
    }

    // ── downsample_buckets ───────────────────────────────────────────────────

    #[test]
    fn test_downsample_buckets_count() {
        let data = pts(&[(0, 1.0), (1000, 2.0)]);
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 2000);
        let ds = TimeBucketAggregator::downsample_buckets(&res);
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn test_downsample_buckets_mean_value() {
        let data = pts(&[(100, 2.0), (200, 4.0)]);
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 1000);
        let ds = TimeBucketAggregator::downsample_buckets(&res);
        assert!((ds[0].value - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_downsample_buckets_timestamp_is_start() {
        let data = pts(&[(500, 1.0)]);
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 1000);
        let ds = TimeBucketAggregator::downsample_buckets(&res);
        assert_eq!(ds[0].timestamp, 0);
    }

    #[test]
    fn test_downsample_empty_bucket_value_zero() {
        let data: Vec<DataPoint> = vec![];
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 1000);
        let ds = TimeBucketAggregator::downsample_buckets(&res);
        assert_eq!(ds[0].value, 0.0);
    }

    // ── bucket alignment ──────────────────────────────────────────────────────

    #[test]
    fn test_aggregate_bucket_start_end_correct() {
        let data: Vec<DataPoint> = vec![];
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 3000);
        assert_eq!(res.buckets[0].start, 0);
        assert_eq!(res.buckets[0].end, 1000);
        assert_eq!(res.buckets[1].start, 1000);
        assert_eq!(res.buckets[2].start, 2000);
    }

    #[test]
    fn test_aggregate_weeks_two_weeks() {
        let one_week = 7 * 86_400_000i64;
        let data = pts(&[(0, 1.0), (one_week, 2.0)]);
        let res =
            TimeBucketAggregator::aggregate(&data, &BucketInterval::Weeks(1), 0, 2 * one_week);
        assert_eq!(res.buckets.len(), 2);
        assert_eq!(res.buckets[0].sum, 1.0);
        assert_eq!(res.buckets[1].sum, 2.0);
    }

    // ── additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_data_point_new() {
        let dp = DataPoint::new(100, 2.71);
        assert_eq!(dp.timestamp, 100);
        assert!((dp.value - 2.71).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bucket_start_zero_interval_uses_one() {
        // zero interval is treated as 1 to avoid division by zero
        let start = TimeBucketAggregator::bucket_start(500, 0);
        // with interval=1 every ts maps to itself
        assert_eq!(start, 500);
    }

    #[test]
    fn test_aggregate_single_bucket_many_points() {
        let data: Vec<DataPoint> = (0..20).map(|i| DataPoint::new(i * 10, i as f64)).collect();
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(500), 0, 500);
        // All 20 points should fit in 1 bucket (timestamps 0..190 < 500).
        assert_eq!(res.buckets.len(), 1);
        assert_eq!(res.buckets[0].count, 20);
    }

    #[test]
    fn test_aggregate_total_points_correct() {
        let data = pts(&[(0, 1.0), (500, 2.0), (1000, 3.0), (1500, 4.0)]);
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 2000);
        assert_eq!(res.total_points, 4);
    }

    #[test]
    fn test_fill_gaps_total_points_includes_filled() {
        let data = pts(&[(0, 1.0)]);
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 3000);
        let filled = TimeBucketAggregator::fill_gaps(&res, 5.0);
        // 1 real + 2 filled = 3 total
        assert_eq!(filled.total_points, 3);
    }

    #[test]
    fn test_downsample_multiple_buckets() {
        let data = pts(&[(0, 4.0), (500, 6.0), (1000, 2.0), (1500, 8.0)]);
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 2000);
        let ds = TimeBucketAggregator::downsample_buckets(&res);
        assert_eq!(ds.len(), 2);
        assert!((ds[0].value - 5.0).abs() < f64::EPSILON); // (4+6)/2
        assert!((ds[1].value - 5.0).abs() < f64::EPSILON); // (2+8)/2
    }

    #[test]
    fn test_aggregate_auto_spreads_into_buckets() {
        let data: Vec<DataPoint> = (0..100)
            .map(|i| DataPoint::new(i as i64 * 1000, i as f64))
            .collect();
        let res = TimeBucketAggregator::aggregate_auto(&data, 20);
        // Every point should be counted.
        assert_eq!(res.total_points, 100);
    }

    #[test]
    fn test_bucket_interval_fixed_equality() {
        assert_eq!(BucketInterval::Fixed(500), BucketInterval::Fixed(500));
        assert_ne!(BucketInterval::Fixed(500), BucketInterval::Fixed(1000));
    }

    #[test]
    fn test_bucket_interval_minutes_two() {
        assert_eq!(BucketInterval::Minutes(2).to_ms(), 120_000);
    }

    #[test]
    fn test_fill_gaps_preserves_count_of_non_empty() {
        let data = pts(&[(0, 10.0), (100, 20.0)]);
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 3000);
        let filled = TimeBucketAggregator::fill_gaps(&res, 0.0);
        assert_eq!(filled.buckets[0].count, 2); // unchanged
    }

    #[test]
    fn test_aggregate_boundary_point_at_start() {
        // A point exactly at `start` should be included.
        let data = pts(&[(0, 42.0)]);
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 1000);
        assert_eq!(res.total_points, 1);
    }

    #[test]
    fn test_aggregate_boundary_point_at_end_excluded() {
        // A point exactly at `end` should NOT be included (range is [start, end)).
        let data = pts(&[(1000, 42.0)]);
        let res = TimeBucketAggregator::aggregate(&data, &BucketInterval::Fixed(1000), 0, 1000);
        assert_eq!(res.total_points, 0);
    }

    #[test]
    fn test_bucket_not_empty_after_point() {
        let mut b = Bucket::new_empty(0, 1000);
        b.push(1.0);
        assert!(!b.is_empty());
    }

    #[test]
    fn test_bucket_min_max_single_point() {
        let mut b = Bucket::new_empty(0, 1000);
        b.push(5.5);
        assert!((b.min - 5.5).abs() < f64::EPSILON);
        assert!((b.max - 5.5).abs() < f64::EPSILON);
    }
}
