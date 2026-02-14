//! Aggregation functions for time-series data
//!
//! Supports common aggregations: AVG, MIN, MAX, SUM, COUNT,
//! as well as statistical functions like STDDEV and VARIANCE.

use crate::error::{TsdbError, TsdbResult};
use crate::series::DataPoint;
use crate::storage::chunks::ChunkMetadata;
use chrono::{DateTime, Utc};

/// Aggregation function type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregation {
    /// Average of values
    Avg,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Sum of values
    Sum,
    /// Count of values
    Count,
    /// First value in range
    First,
    /// Last value in range
    Last,
    /// Standard deviation
    StdDev,
    /// Variance
    Variance,
    /// Median value
    Median,
    /// Percentile (0-100)
    Percentile(u8),
}

/// Result of an aggregation query
#[derive(Debug, Clone)]
pub struct AggregationResult {
    /// Series ID
    pub series_id: u64,
    /// Aggregation function used
    pub aggregation: Aggregation,
    /// Start of aggregation window
    pub start_time: DateTime<Utc>,
    /// End of aggregation window
    pub end_time: DateTime<Utc>,
    /// Aggregated value
    pub value: f64,
    /// Number of data points aggregated
    pub count: usize,
}

/// Aggregation calculator
#[derive(Debug, Default)]
pub struct Aggregator {
    count: usize,
    sum: f64,
    min: f64,
    max: f64,
    first: Option<f64>,
    last: Option<f64>,
    first_time: Option<DateTime<Utc>>,
    last_time: Option<DateTime<Utc>>,
    // For variance/stddev calculation (Welford's algorithm)
    mean: f64,
    m2: f64, // Sum of squared differences from mean
    // For median/percentile
    values: Vec<f64>,
}

impl Aggregator {
    /// Create a new aggregator
    pub fn new() -> Self {
        Self {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            ..Default::default()
        }
    }

    /// Add a data point to the aggregation
    pub fn add(&mut self, point: &DataPoint) {
        self.count += 1;
        self.sum += point.value;

        // Min/Max
        if point.value < self.min {
            self.min = point.value;
        }
        if point.value > self.max {
            self.max = point.value;
        }

        // First/Last
        if self.first.is_none() {
            self.first = Some(point.value);
            self.first_time = Some(point.timestamp);
        }
        self.last = Some(point.value);
        self.last_time = Some(point.timestamp);

        // Welford's algorithm for online variance
        let delta = point.value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = point.value - self.mean;
        self.m2 += delta * delta2;

        // Store for median/percentile
        self.values.push(point.value);
    }

    /// Add a batch of data points
    pub fn add_batch(&mut self, points: &[DataPoint]) {
        for point in points {
            self.add(point);
        }
    }

    /// Use chunk metadata for optimized aggregation (when possible)
    ///
    /// This avoids decompression for MIN/MAX/COUNT/AVG queries
    pub fn add_from_metadata(&mut self, metadata: &ChunkMetadata) {
        // Can only use for simple aggregations
        // Note: This is an approximation for AVG across multiple chunks
        self.count += metadata.count;
        self.sum += metadata.avg_value * metadata.count as f64;

        if metadata.min_value < self.min {
            self.min = metadata.min_value;
        }
        if metadata.max_value > self.max {
            self.max = metadata.max_value;
        }
    }

    /// Calculate aggregation result
    pub fn result(&mut self, aggregation: Aggregation) -> TsdbResult<f64> {
        if self.count == 0 {
            return Err(TsdbError::Query(
                "No data points for aggregation".to_string(),
            ));
        }

        match aggregation {
            Aggregation::Avg => Ok(self.sum / self.count as f64),
            Aggregation::Min => Ok(self.min),
            Aggregation::Max => Ok(self.max),
            Aggregation::Sum => Ok(self.sum),
            Aggregation::Count => Ok(self.count as f64),
            Aggregation::First => self
                .first
                .ok_or_else(|| TsdbError::Query("No first value".to_string())),
            Aggregation::Last => self
                .last
                .ok_or_else(|| TsdbError::Query("No last value".to_string())),
            Aggregation::StdDev => {
                if self.count < 2 {
                    return Ok(0.0);
                }
                Ok((self.m2 / (self.count - 1) as f64).sqrt())
            }
            Aggregation::Variance => {
                if self.count < 2 {
                    return Ok(0.0);
                }
                Ok(self.m2 / (self.count - 1) as f64)
            }
            Aggregation::Median => self.percentile_result(50),
            Aggregation::Percentile(p) => self.percentile_result(p),
        }
    }

    /// Calculate percentile
    fn percentile_result(&mut self, percentile: u8) -> TsdbResult<f64> {
        if self.values.is_empty() {
            return Err(TsdbError::Query("No values for percentile".to_string()));
        }

        // Sort values
        self.values
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = (percentile as f64 / 100.0 * (self.values.len() - 1) as f64).round() as usize;
        Ok(self.values[idx.min(self.values.len() - 1)])
    }

    /// Get the count of aggregated points
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get the start time (first point timestamp)
    pub fn start_time(&self) -> Option<DateTime<Utc>> {
        self.first_time
    }

    /// Get the end time (last point timestamp)
    pub fn end_time(&self) -> Option<DateTime<Utc>> {
        self.last_time
    }
}

/// Compute multiple aggregations at once (more efficient)
pub fn compute_aggregations(
    points: &[DataPoint],
    aggregations: &[Aggregation],
) -> TsdbResult<Vec<f64>> {
    if points.is_empty() {
        return Err(TsdbError::Query("No data points".to_string()));
    }

    let mut aggregator = Aggregator::new();
    aggregator.add_batch(points);

    aggregations
        .iter()
        .map(|agg| aggregator.result(*agg))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

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
    fn test_aggregation_avg() {
        let now = Utc::now();
        let points = create_test_points(now, &[10.0, 20.0, 30.0, 40.0, 50.0]);

        let mut agg = Aggregator::new();
        agg.add_batch(&points);

        let result = agg.result(Aggregation::Avg).expect("result should be Ok");
        assert!((result - 30.0).abs() < 0.001);
    }

    #[test]
    fn test_aggregation_min_max() {
        let now = Utc::now();
        let points = create_test_points(now, &[10.0, 5.0, 30.0, 15.0, 25.0]);

        let mut agg = Aggregator::new();
        agg.add_batch(&points);

        assert!(
            (agg.result(Aggregation::Min)
                .expect("operation should succeed")
                - 5.0)
                .abs()
                < 0.001
        );
        assert!(
            (agg.result(Aggregation::Max)
                .expect("operation should succeed")
                - 30.0)
                .abs()
                < 0.001
        );
    }

    #[test]
    fn test_aggregation_sum_count() {
        let now = Utc::now();
        let points = create_test_points(now, &[10.0, 20.0, 30.0]);

        let mut agg = Aggregator::new();
        agg.add_batch(&points);

        assert!(
            (agg.result(Aggregation::Sum)
                .expect("operation should succeed")
                - 60.0)
                .abs()
                < 0.001
        );
        assert!(
            (agg.result(Aggregation::Count)
                .expect("operation should succeed")
                - 3.0)
                .abs()
                < 0.001
        );
    }

    #[test]
    fn test_aggregation_first_last() {
        let now = Utc::now();
        let points = create_test_points(now, &[10.0, 20.0, 30.0]);

        let mut agg = Aggregator::new();
        agg.add_batch(&points);

        assert!(
            (agg.result(Aggregation::First)
                .expect("operation should succeed")
                - 10.0)
                .abs()
                < 0.001
        );
        assert!(
            (agg.result(Aggregation::Last)
                .expect("operation should succeed")
                - 30.0)
                .abs()
                < 0.001
        );
    }

    #[test]
    fn test_aggregation_stddev() {
        let now = Utc::now();
        // Values: 2, 4, 4, 4, 5, 5, 7, 9
        // Mean = 5, StdDev ≈ 2.0
        let points = create_test_points(now, &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);

        let mut agg = Aggregator::new();
        agg.add_batch(&points);

        let stddev = agg
            .result(Aggregation::StdDev)
            .expect("result should be Ok");
        assert!((stddev - 2.138).abs() < 0.01);
    }

    #[test]
    fn test_aggregation_median() {
        let now = Utc::now();
        let points = create_test_points(now, &[1.0, 3.0, 5.0, 7.0, 9.0]);

        let mut agg = Aggregator::new();
        agg.add_batch(&points);

        let median = agg
            .result(Aggregation::Median)
            .expect("result should be Ok");
        assert!((median - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_aggregation_percentile() {
        let now = Utc::now();
        let points = create_test_points(now, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        let mut agg = Aggregator::new();
        agg.add_batch(&points);

        // 90th percentile should be around 9.0
        let p90 = agg
            .result(Aggregation::Percentile(90))
            .expect("result should be Ok");
        assert!((p90 - 9.0).abs() < 1.0);
    }

    #[test]
    fn test_compute_multiple_aggregations() {
        let now = Utc::now();
        let points = create_test_points(now, &[10.0, 20.0, 30.0, 40.0, 50.0]);

        let results = compute_aggregations(
            &points,
            &[Aggregation::Avg, Aggregation::Min, Aggregation::Max],
        )
        .expect("operation should succeed");

        assert!((results[0] - 30.0).abs() < 0.001); // AVG
        assert!((results[1] - 10.0).abs() < 0.001); // MIN
        assert!((results[2] - 50.0).abs() < 0.001); // MAX
    }
}
