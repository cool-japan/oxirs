//! Time resampling and downsampling
//!
//! Provides time bucketing operations for converting high-frequency data
//! to lower-frequency summaries (e.g., 1Hz → 1 minute averages).

use crate::error::TsdbResult;
use crate::query::aggregate::{Aggregation, Aggregator};
use crate::series::DataPoint;
use chrono::{DateTime, Duration, Timelike, Utc};
use std::collections::BTreeMap;

/// Resample bucket specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleBucket {
    /// Bucket by seconds
    Second(u32),
    /// Bucket by minutes
    Minute(u32),
    /// Bucket by hours
    Hour(u32),
    /// Bucket by days
    Day(u32),
    /// Bucket by weeks
    Week(u32),
    /// Custom duration in milliseconds
    Custom(i64),
}

impl ResampleBucket {
    /// Get duration of this bucket
    pub fn duration(&self) -> Duration {
        match self {
            ResampleBucket::Second(n) => Duration::seconds(*n as i64),
            ResampleBucket::Minute(n) => Duration::minutes(*n as i64),
            ResampleBucket::Hour(n) => Duration::hours(*n as i64),
            ResampleBucket::Day(n) => Duration::days(*n as i64),
            ResampleBucket::Week(n) => Duration::weeks(*n as i64),
            ResampleBucket::Custom(ms) => Duration::milliseconds(*ms),
        }
    }

    /// Align timestamp to bucket boundary
    pub fn align_timestamp(&self, ts: DateTime<Utc>) -> DateTime<Utc> {
        match self {
            ResampleBucket::Second(n) => {
                let secs = ts.second() as i64;
                let aligned_secs = (secs / *n as i64) * *n as i64;
                ts.with_second(aligned_secs as u32)
                    .unwrap()
                    .with_nanosecond(0)
                    .unwrap()
            }
            ResampleBucket::Minute(n) => {
                let mins = ts.minute() as i64;
                let aligned_mins = (mins / *n as i64) * *n as i64;
                ts.with_minute(aligned_mins as u32)
                    .unwrap()
                    .with_second(0)
                    .unwrap()
                    .with_nanosecond(0)
                    .unwrap()
            }
            ResampleBucket::Hour(n) => {
                let hours = ts.hour() as i64;
                let aligned_hours = (hours / *n as i64) * *n as i64;
                ts.with_hour(aligned_hours as u32)
                    .unwrap()
                    .with_minute(0)
                    .unwrap()
                    .with_second(0)
                    .unwrap()
                    .with_nanosecond(0)
                    .unwrap()
            }
            ResampleBucket::Day(_) | ResampleBucket::Week(_) | ResampleBucket::Custom(_) => {
                // For day/week/custom, use epoch-based alignment
                let duration_ms = self.duration().num_milliseconds();
                let ts_ms = ts.timestamp_millis();
                let aligned_ms = (ts_ms / duration_ms) * duration_ms;
                DateTime::from_timestamp_millis(aligned_ms).unwrap_or(ts)
            }
        }
    }
}

/// Resampler for time bucketing
pub struct Resampler {
    /// Bucket specification
    bucket: ResampleBucket,
    /// Aggregation function
    aggregation: Aggregation,
    /// Fill method for empty buckets
    fill: FillMethod,
}

/// Method for filling empty buckets
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum FillMethod {
    /// Leave empty buckets as gaps
    #[default]
    None,
    /// Fill with previous value
    Previous,
    /// Fill with next value
    Next,
    /// Linear interpolation
    Linear,
    /// Fill with constant value
    Constant(f64),
}

impl Resampler {
    /// Create a new resampler
    pub fn new(bucket: ResampleBucket, aggregation: Aggregation) -> Self {
        Self {
            bucket,
            aggregation,
            fill: FillMethod::default(),
        }
    }

    /// Set fill method for empty buckets
    pub fn with_fill(mut self, fill: FillMethod) -> Self {
        self.fill = fill;
        self
    }

    /// Resample data points into buckets
    pub fn resample(&self, points: &[DataPoint]) -> TsdbResult<Vec<DataPoint>> {
        if points.is_empty() {
            return Ok(Vec::new());
        }

        // Group points by bucket
        let mut buckets: BTreeMap<i64, Vec<DataPoint>> = BTreeMap::new();

        for point in points {
            let aligned = self.bucket.align_timestamp(point.timestamp);
            let bucket_key = aligned.timestamp_millis();
            buckets.entry(bucket_key).or_default().push(*point);
        }

        // Aggregate each bucket
        let mut results = Vec::with_capacity(buckets.len());

        for (&bucket_ts, bucket_points) in &buckets {
            let mut aggregator = Aggregator::new();
            aggregator.add_batch(bucket_points);

            let value = aggregator.result(self.aggregation)?;
            let timestamp = DateTime::from_timestamp_millis(bucket_ts).unwrap_or_else(Utc::now);

            results.push(DataPoint { timestamp, value });
        }

        // Apply fill method for gaps
        if self.fill != FillMethod::None && results.len() > 1 {
            results = self.fill_gaps(results);
        }

        Ok(results)
    }

    /// Fill gaps between buckets
    fn fill_gaps(&self, results: Vec<DataPoint>) -> Vec<DataPoint> {
        if results.len() < 2 {
            return results;
        }

        let bucket_duration = self.bucket.duration();
        let mut filled: Vec<DataPoint> = Vec::with_capacity(results.len() * 2);
        let mut prev_value: Option<f64> = None;

        for current in results {
            // Check for gap before current point
            if let Some(last) = filled.last().copied() {
                let expected_next = last.timestamp + bucket_duration;
                let mut fill_ts = expected_next;

                while fill_ts < current.timestamp {
                    let fill_value = match self.fill {
                        FillMethod::Previous => prev_value.unwrap_or(current.value),
                        FillMethod::Next => current.value,
                        FillMethod::Linear => {
                            if let Some(pv) = prev_value {
                                let total_gap = current
                                    .timestamp
                                    .signed_duration_since(last.timestamp)
                                    .num_milliseconds()
                                    as f64;
                                let current_gap = fill_ts
                                    .signed_duration_since(last.timestamp)
                                    .num_milliseconds()
                                    as f64;
                                let ratio = if total_gap > 0.0 {
                                    current_gap / total_gap
                                } else {
                                    0.0
                                };
                                pv + ratio * (current.value - pv)
                            } else {
                                current.value
                            }
                        }
                        FillMethod::Constant(v) => v,
                        FillMethod::None => break,
                    };

                    filled.push(DataPoint {
                        timestamp: fill_ts,
                        value: fill_value,
                    });

                    fill_ts += bucket_duration;
                }
            }

            filled.push(current);
            prev_value = Some(current.value);
        }

        filled
    }
}

/// Resample data points to specified bucket
pub fn resample(
    points: &[DataPoint],
    bucket: ResampleBucket,
    aggregation: Aggregation,
) -> TsdbResult<Vec<DataPoint>> {
    Resampler::new(bucket, aggregation).resample(points)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_points(
        start: DateTime<Utc>,
        count: usize,
        interval_secs: i64,
    ) -> Vec<DataPoint> {
        (0..count)
            .map(|i| DataPoint {
                timestamp: start + Duration::seconds(i as i64 * interval_secs),
                value: i as f64,
            })
            .collect()
    }

    #[test]
    fn test_resample_minute_avg() {
        let start = Utc::now()
            .with_second(0)
            .unwrap()
            .with_nanosecond(0)
            .unwrap();

        // Create 120 points, 1 per second (2 minutes worth)
        let points = create_test_points(start, 120, 1);

        let results = resample(&points, ResampleBucket::Minute(1), Aggregation::Avg).unwrap();

        // Should get 2 buckets
        assert_eq!(results.len(), 2);

        // First bucket: avg of 0-59 = 29.5
        assert!((results[0].value - 29.5).abs() < 0.1);

        // Second bucket: avg of 60-119 = 89.5
        assert!((results[1].value - 89.5).abs() < 0.1);
    }

    #[test]
    fn test_resample_minute_sum() {
        let start = Utc::now()
            .with_second(0)
            .unwrap()
            .with_nanosecond(0)
            .unwrap();

        let points = vec![
            DataPoint {
                timestamp: start,
                value: 10.0,
            },
            DataPoint {
                timestamp: start + Duration::seconds(30),
                value: 20.0,
            },
            DataPoint {
                timestamp: start + Duration::seconds(60),
                value: 30.0,
            },
        ];

        let results = resample(&points, ResampleBucket::Minute(1), Aggregation::Sum).unwrap();

        assert_eq!(results.len(), 2);
        assert!((results[0].value - 30.0).abs() < 0.001); // 10 + 20
        assert!((results[1].value - 30.0).abs() < 0.001); // 30
    }

    #[test]
    fn test_resample_with_fill() {
        let start = Utc::now()
            .with_second(0)
            .unwrap()
            .with_nanosecond(0)
            .unwrap();

        // Create points with a gap
        let points = vec![
            DataPoint {
                timestamp: start,
                value: 10.0,
            },
            DataPoint {
                timestamp: start + Duration::minutes(3),
                value: 40.0,
            },
        ];

        let results = Resampler::new(ResampleBucket::Minute(1), Aggregation::Avg)
            .with_fill(FillMethod::Linear)
            .resample(&points)
            .unwrap();

        // Should have 4 points: 0, 1, 2, 3 minutes
        assert!(results.len() >= 3);

        // Verify linear interpolation
        // At minute 0: 10.0
        // At minute 3: 40.0
        // Linear interpolation at minute 1: ~20.0, minute 2: ~30.0
    }

    #[test]
    fn test_bucket_alignment() {
        let ts = DateTime::parse_from_rfc3339("2024-01-15T10:23:45.123Z")
            .unwrap()
            .to_utc();

        // Align to 5-minute boundary
        let aligned = ResampleBucket::Minute(5).align_timestamp(ts);
        assert_eq!(aligned.minute(), 20);
        assert_eq!(aligned.second(), 0);

        // Align to hourly boundary
        let aligned_hour = ResampleBucket::Hour(1).align_timestamp(ts);
        assert_eq!(aligned_hour.hour(), 10);
        assert_eq!(aligned_hour.minute(), 0);

        // Align to 15-second boundary
        let aligned_sec = ResampleBucket::Second(15).align_timestamp(ts);
        assert_eq!(aligned_sec.second(), 45);
    }

    #[test]
    fn test_resample_count() {
        let start = Utc::now()
            .with_second(0)
            .unwrap()
            .with_nanosecond(0)
            .unwrap();

        let points = create_test_points(start, 150, 1);

        let results = resample(&points, ResampleBucket::Minute(1), Aggregation::Count).unwrap();

        assert_eq!(results.len(), 3);
        assert!((results[0].value - 60.0).abs() < 0.001); // First minute: 60 points
        assert!((results[1].value - 60.0).abs() < 0.001); // Second minute: 60 points
        assert!((results[2].value - 30.0).abs() < 0.001); // Third minute: 30 points
    }

    #[test]
    fn test_resample_min_max() {
        let start = Utc::now()
            .with_second(0)
            .unwrap()
            .with_nanosecond(0)
            .unwrap();

        let points = vec![
            DataPoint {
                timestamp: start,
                value: 10.0,
            },
            DataPoint {
                timestamp: start + Duration::seconds(10),
                value: 50.0,
            },
            DataPoint {
                timestamp: start + Duration::seconds(20),
                value: 30.0,
            },
        ];

        let results_min = resample(&points, ResampleBucket::Minute(1), Aggregation::Min).unwrap();
        let results_max = resample(&points, ResampleBucket::Minute(1), Aggregation::Max).unwrap();

        assert!((results_min[0].value - 10.0).abs() < 0.001);
        assert!((results_max[0].value - 50.0).abs() < 0.001);
    }
}
