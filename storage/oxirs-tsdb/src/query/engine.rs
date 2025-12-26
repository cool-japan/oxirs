//! Query engine for time-series data
//!
//! Provides a fluent API for building and executing queries.

use crate::error::{TsdbError, TsdbResult};
use crate::query::aggregate::{Aggregation, Aggregator};
use crate::query::interpolate::{InterpolateMethod, Interpolator};
use crate::query::range::TimeRange;
use crate::query::resample::ResampleBucket;
use crate::query::window::WindowSpec;
use crate::series::DataPoint;
use crate::storage::TimeChunk;
use chrono::{DateTime, Duration, Utc};

/// Query result
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Series ID
    pub series_id: u64,
    /// Result data points
    pub points: Vec<DataPoint>,
    /// Aggregated value (if aggregation was applied)
    pub aggregated_value: Option<f64>,
    /// Query execution time
    pub execution_time_ms: u64,
    /// Number of chunks scanned
    pub chunks_scanned: usize,
    /// Total points processed
    pub points_processed: usize,
}

/// Query engine for executing time-series queries
#[derive(Debug, Default)]
pub struct QueryEngine {
    /// All loaded chunks (in a real impl, this would be a storage backend)
    chunks: Vec<TimeChunk>,
}

impl QueryEngine {
    /// Create a new query engine
    pub fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    /// Add a chunk to the engine
    pub fn add_chunk(&mut self, chunk: TimeChunk) {
        self.chunks.push(chunk);
    }

    /// Add multiple chunks
    pub fn add_chunks(&mut self, chunks: Vec<TimeChunk>) {
        self.chunks.extend(chunks);
    }

    /// Start building a query
    pub fn query(&self) -> QueryBuilder<'_> {
        QueryBuilder::new(self)
    }

    /// Get chunk metadata for optimization
    pub fn get_chunks_for_series(&self, series_id: u64) -> Vec<&TimeChunk> {
        self.chunks
            .iter()
            .filter(|c| c.series_id == series_id)
            .collect()
    }
}

/// Builder for constructing queries
pub struct QueryBuilder<'a> {
    engine: &'a QueryEngine,
    series_id: Option<u64>,
    time_range: Option<TimeRange>,
    aggregation: Option<Aggregation>,
    window: Option<WindowSpec>,
    resample: Option<ResampleBucket>,
    interpolate: Option<InterpolateMethod>,
    limit: Option<usize>,
    order_descending: bool,
}

impl<'a> QueryBuilder<'a> {
    /// Create a new query builder
    fn new(engine: &'a QueryEngine) -> Self {
        Self {
            engine,
            series_id: None,
            time_range: None,
            aggregation: None,
            window: None,
            resample: None,
            interpolate: None,
            limit: None,
            order_descending: false,
        }
    }

    /// Set the series to query
    pub fn series(mut self, series_id: u64) -> Self {
        self.series_id = Some(series_id);
        self
    }

    /// Set the time range
    pub fn time_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.time_range = Some(TimeRange::new(start, end));
        self
    }

    /// Set time range relative to now
    pub fn last(mut self, duration: Duration) -> Self {
        let now = Utc::now();
        self.time_range = Some(TimeRange::new(now - duration, now));
        self
    }

    /// Apply aggregation
    pub fn aggregate(mut self, agg: Aggregation) -> Self {
        self.aggregation = Some(agg);
        self
    }

    /// Apply window function
    pub fn window(mut self, spec: WindowSpec) -> Self {
        self.window = Some(spec);
        self
    }

    /// Apply resampling
    pub fn resample(mut self, bucket: ResampleBucket) -> Self {
        self.resample = Some(bucket);
        self
    }

    /// Apply interpolation
    pub fn interpolate(mut self, method: InterpolateMethod) -> Self {
        self.interpolate = Some(method);
        self
    }

    /// Limit number of results
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Order results descending (newest first)
    pub fn descending(mut self) -> Self {
        self.order_descending = true;
        self
    }

    /// Execute the query
    pub fn execute(self) -> TsdbResult<QueryResult> {
        let start_time = std::time::Instant::now();

        let series_id = self
            .series_id
            .ok_or_else(|| TsdbError::Query("Series ID required".to_string()))?;

        // Get relevant chunks
        let chunks: Vec<&TimeChunk> = self
            .engine
            .chunks
            .iter()
            .filter(|c| {
                c.series_id == series_id
                    && self
                        .time_range
                        .as_ref()
                        .map(|tr| tr.overlaps_chunk(c))
                        .unwrap_or(true)
            })
            .collect();

        let chunks_scanned = chunks.len();

        // Decompress and filter data
        let mut points: Vec<DataPoint> = Vec::new();

        for chunk in &chunks {
            let chunk_points = if let Some(ref range) = self.time_range {
                chunk.query_range(range.start, range.end)?
            } else {
                chunk.decompress()?
            };
            points.extend(chunk_points);
        }

        let points_processed = points.len();

        // Sort by timestamp
        points.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        // Apply window function if specified
        if let Some(spec) = self.window {
            let mut calculator = crate::query::window::WindowCalculator::new(spec);
            points = calculator.apply(&points);
        }

        // Apply resampling if specified
        if let Some(bucket) = self.resample {
            let resampler = crate::query::resample::Resampler::new(
                bucket,
                self.aggregation.unwrap_or(Aggregation::Avg),
            );
            points = resampler.resample(&points)?;
        }

        // Apply interpolation if specified
        if let Some(method) = self.interpolate {
            let interpolator = Interpolator::new(method);
            if let Some(ref range) = self.time_range {
                // Fill at 1-second intervals by default
                points = interpolator.fill_at_interval(
                    &points,
                    Duration::seconds(1),
                    Some(range.start),
                    Some(range.end),
                )?;
            }
        }

        // Calculate aggregation if specified (and no resampling)
        let aggregated_value = if self.aggregation.is_some() && self.resample.is_none() {
            let mut aggregator = Aggregator::new();
            aggregator.add_batch(&points);
            Some(aggregator.result(self.aggregation.unwrap())?)
        } else {
            None
        };

        // Apply ordering
        if self.order_descending {
            points.reverse();
        }

        // Apply limit
        if let Some(limit) = self.limit {
            points.truncate(limit);
        }

        let execution_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(QueryResult {
            series_id,
            points,
            aggregated_value,
            execution_time_ms,
            chunks_scanned,
            points_processed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::window::WindowFunction;

    fn create_test_engine() -> QueryEngine {
        let mut engine = QueryEngine::new();
        let start = Utc::now();

        // Create test chunk with 100 points
        let points: Vec<DataPoint> = (0..100)
            .map(|i| DataPoint {
                timestamp: start + Duration::seconds(i),
                value: i as f64,
            })
            .collect();

        let chunk = TimeChunk::new(1, start, Duration::hours(2), points).unwrap();
        engine.add_chunk(chunk);

        engine
    }

    #[test]
    fn test_basic_query() {
        let engine = create_test_engine();

        let result = engine.query().series(1).execute().unwrap();

        assert_eq!(result.series_id, 1);
        assert_eq!(result.points.len(), 100);
        assert_eq!(result.chunks_scanned, 1);
    }

    #[test]
    fn test_query_with_time_range() {
        let engine = create_test_engine();
        let now = Utc::now();

        let result = engine
            .query()
            .series(1)
            .time_range(now + Duration::seconds(20), now + Duration::seconds(30))
            .execute()
            .unwrap();

        // Should get 10 points (20-29)
        assert_eq!(result.points.len(), 10);
    }

    #[test]
    fn test_query_with_aggregation() {
        let engine = create_test_engine();

        let result = engine
            .query()
            .series(1)
            .aggregate(Aggregation::Avg)
            .execute()
            .unwrap();

        // Average of 0-99 = 49.5
        let avg = result.aggregated_value.unwrap();
        assert!((avg - 49.5).abs() < 0.1);
    }

    #[test]
    fn test_query_with_limit() {
        let engine = create_test_engine();

        let result = engine.query().series(1).limit(10).execute().unwrap();

        assert_eq!(result.points.len(), 10);
    }

    #[test]
    fn test_query_descending() {
        let engine = create_test_engine();

        let result = engine
            .query()
            .series(1)
            .descending()
            .limit(5)
            .execute()
            .unwrap();

        // Should be in descending order
        for window in result.points.windows(2) {
            assert!(window[0].timestamp >= window[1].timestamp);
        }
    }

    #[test]
    fn test_query_with_window() {
        let engine = create_test_engine();

        let result = engine
            .query()
            .series(1)
            .window(WindowSpec::count_based(5, WindowFunction::MovingAverage))
            .execute()
            .unwrap();

        // Moving average reduces points by window size - 1
        assert!(result.points.len() <= 100);
    }

    #[test]
    fn test_last_duration() {
        let mut engine = QueryEngine::new();
        let now = Utc::now();

        // Create points in the last hour
        let points: Vec<DataPoint> = (0..60)
            .map(|i| DataPoint {
                timestamp: now - Duration::minutes(i as i64),
                value: i as f64,
            })
            .collect();

        let chunk =
            TimeChunk::new(1, now - Duration::hours(1), Duration::hours(2), points).unwrap();
        engine.add_chunk(chunk);

        let result = engine
            .query()
            .series(1)
            .last(Duration::minutes(30))
            .execute()
            .unwrap();

        // Should get approximately 30 points
        assert!(result.points.len() <= 35);
    }

    #[test]
    fn test_query_nonexistent_series() {
        let engine = create_test_engine();

        let result = engine.query().series(999).execute().unwrap();

        // No chunks for series 999
        assert_eq!(result.chunks_scanned, 0);
        assert!(result.points.is_empty());
    }
}
