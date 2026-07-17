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
use crate::storage::{ColumnarStore, TimeChunk};
use chrono::{DateTime, Duration, Utc};
use std::sync::Arc;

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

/// Backing storage for a [`QueryEngine`].
///
/// `InMemory` keeps chunks fully resident in a `Vec`, as loaded by
/// [`QueryEngine::add_chunk`]/[`QueryEngine::add_chunks`]. `Columnar` instead
/// reads chunk metadata from the [`crate::storage::SeriesIndex`] and lazily
/// decompresses only the chunks that overlap the requested time range,
/// backed by the crate's real on-disk [`ColumnarStore`] pipeline.
#[derive(Debug)]
enum ChunkSource {
    InMemory(Vec<TimeChunk>),
    Columnar(Arc<ColumnarStore>),
}

/// Query engine for executing time-series queries
#[derive(Debug)]
pub struct QueryEngine {
    source: ChunkSource,
}

impl Default for QueryEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryEngine {
    /// Create a new, empty in-memory query engine.
    ///
    /// Chunks must be loaded explicitly via [`QueryEngine::add_chunk`] /
    /// [`QueryEngine::add_chunks`]. For querying a durable, disk-backed
    /// series without pre-loading every chunk into memory, use
    /// [`QueryEngine::from_columnar_store`] instead.
    pub fn new() -> Self {
        Self {
            source: ChunkSource::InMemory(Vec::new()),
        }
    }

    /// Create a query engine that lazily reads chunks from a durable
    /// [`ColumnarStore`] by time range instead of requiring the whole
    /// series to be pre-loaded into memory.
    ///
    /// Only the chunks overlapping a query's requested time range (or, when
    /// no range is given, all chunks for the series) are read from disk and
    /// decompressed; nothing is cached beyond `ColumnarStore`'s own chunk
    /// cache.
    pub fn from_columnar_store(store: Arc<ColumnarStore>) -> Self {
        Self {
            source: ChunkSource::Columnar(store),
        }
    }

    /// True if this engine reads lazily from a [`ColumnarStore`] rather than
    /// holding chunks in memory.
    pub fn is_columnar_backed(&self) -> bool {
        matches!(self.source, ChunkSource::Columnar(_))
    }

    /// Add a chunk to the engine.
    ///
    /// Only meaningful for an in-memory engine (created via
    /// [`QueryEngine::new`]); on a [`QueryEngine::from_columnar_store`]
    /// engine this is a no-op since chunks are read lazily from disk --
    /// write through [`ColumnarStore::write_chunk`] directly instead.
    pub fn add_chunk(&mut self, chunk: TimeChunk) {
        if let ChunkSource::InMemory(chunks) = &mut self.source {
            chunks.push(chunk);
        } else {
            tracing::warn!(
                series_id = chunk.series_id,
                "add_chunk() is a no-op on a ColumnarStore-backed QueryEngine; \
                 write through ColumnarStore::write_chunk() instead"
            );
        }
    }

    /// Add multiple chunks. See [`QueryEngine::add_chunk`] for the
    /// ColumnarStore-backed caveat.
    pub fn add_chunks(&mut self, chunks: Vec<TimeChunk>) {
        for chunk in chunks {
            self.add_chunk(chunk);
        }
    }

    /// Start building a query
    pub fn query(&self) -> QueryBuilder<'_> {
        QueryBuilder::new(self)
    }

    /// Get chunk metadata for optimization.
    ///
    /// Only populated for an in-memory engine; a [`QueryEngine::from_columnar_store`]
    /// engine does not hold decompressed `TimeChunk`s in memory and always
    /// returns an empty list here (use its `SeriesIndex` directly for chunk
    /// metadata without decompression).
    pub fn get_chunks_for_series(&self, series_id: u64) -> Vec<&TimeChunk> {
        match &self.source {
            ChunkSource::InMemory(chunks) => {
                chunks.iter().filter(|c| c.series_id == series_id).collect()
            }
            ChunkSource::Columnar(_) => Vec::new(),
        }
    }

    /// Load the data points for `series_id` (optionally restricted to
    /// `time_range`) along with the number of chunks scanned to produce
    /// them, regardless of backing storage.
    fn load_points(
        &self,
        series_id: u64,
        time_range: Option<&TimeRange>,
    ) -> TsdbResult<(Vec<DataPoint>, usize)> {
        match &self.source {
            ChunkSource::InMemory(chunks) => {
                let relevant: Vec<&TimeChunk> = chunks
                    .iter()
                    .filter(|c| {
                        c.series_id == series_id
                            && time_range.map(|tr| tr.overlaps_chunk(c)).unwrap_or(true)
                    })
                    .collect();

                let chunks_scanned = relevant.len();
                let mut points: Vec<DataPoint> = Vec::new();
                for chunk in relevant {
                    let chunk_points = if let Some(range) = time_range {
                        chunk.query_range(range.start, range.end)?
                    } else {
                        chunk.decompress()?
                    };
                    points.extend(chunk_points);
                }
                Ok((points, chunks_scanned))
            }
            ChunkSource::Columnar(store) => {
                let entries = match time_range {
                    Some(range) => {
                        store
                            .index()
                            .get_chunks_in_range(series_id, range.start, range.end)?
                    }
                    None => store.index().get_chunks_for_series(series_id)?,
                };

                let chunks_scanned = entries.len();
                let mut points: Vec<DataPoint> = Vec::new();
                for entry in &entries {
                    let chunk = store.read_chunk(entry.chunk_id)?;
                    let chunk_points = if let Some(range) = time_range {
                        chunk.query_range(range.start, range.end)?
                    } else {
                        chunk.decompress()?
                    };
                    points.extend(chunk_points);
                }
                points.sort_by_key(|p| p.timestamp);
                Ok((points, chunks_scanned))
            }
        }
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

        // Read the relevant points lazily from whichever backing storage
        // this engine uses (in-memory `Vec<TimeChunk>` or a durable
        // `ColumnarStore`, read by time range via its `SeriesIndex`).
        let (mut points, chunks_scanned) = self
            .engine
            .load_points(series_id, self.time_range.as_ref())?;

        let points_processed = points.len();

        // Sort by timestamp
        points.sort_by_key(|p| p.timestamp);

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
        let aggregated_value = if let (Some(agg), None) = (self.aggregation, self.resample) {
            let mut aggregator = Aggregator::new();
            aggregator.add_batch(&points);
            Some(aggregator.result(agg)?)
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

        let chunk = TimeChunk::new(1, start, Duration::hours(2), points)
            .expect("construction should succeed");
        engine.add_chunk(chunk);

        engine
    }

    #[test]
    fn test_basic_query() {
        let engine = create_test_engine();

        let result = engine
            .query()
            .series(1)
            .execute()
            .expect("query should succeed");

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
            .expect("operation should succeed");

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
            .expect("operation should succeed");

        // Average of 0-99 = 49.5
        let avg = result.aggregated_value.expect("result should be Ok");
        assert!((avg - 49.5).abs() < 0.1);
    }

    #[test]
    fn test_query_with_limit() {
        let engine = create_test_engine();

        let result = engine
            .query()
            .series(1)
            .limit(10)
            .execute()
            .expect("query should succeed");

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
            .expect("operation should succeed");

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
            .expect("operation should succeed");

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

        let chunk = TimeChunk::new(1, now - Duration::hours(1), Duration::hours(2), points)
            .expect("chunk operation should succeed");
        engine.add_chunk(chunk);

        let result = engine
            .query()
            .series(1)
            .last(Duration::minutes(30))
            .execute()
            .expect("operation should succeed");

        // Should get approximately 30 points
        assert!(result.points.len() <= 35);
    }

    #[test]
    fn test_query_nonexistent_series() {
        let engine = create_test_engine();

        let result = engine
            .query()
            .series(999)
            .execute()
            .expect("query should succeed");

        // No chunks for series 999
        assert_eq!(result.chunks_scanned, 0);
        assert!(result.points.is_empty());
    }

    // -- ColumnarStore-backed engine (regression tests for the P1 finding) --

    fn temp_columnar_store(name: &str) -> (Arc<ColumnarStore>, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "oxirs_tsdb_query_engine_test_{name}_{}",
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let _ = std::fs::remove_dir_all(&path);
        let mut store = ColumnarStore::new(&path, Duration::hours(2), 16)
            .expect("columnar store creation should succeed");
        store.set_fsync(false);
        (Arc::new(store), path)
    }

    #[test]
    fn test_columnar_backed_engine_reads_lazily_from_disk() {
        let (store, path) = temp_columnar_store("basic");
        let start = Utc::now();

        let points: Vec<DataPoint> = (0..100)
            .map(|i| DataPoint {
                timestamp: start + Duration::seconds(i),
                value: i as f64,
            })
            .collect();
        let chunk = TimeChunk::new(1, start, Duration::hours(2), points)
            .expect("chunk construction should succeed");
        store
            .write_chunk(&chunk)
            .expect("chunk write should succeed");

        let engine = QueryEngine::from_columnar_store(Arc::clone(&store));
        assert!(engine.is_columnar_backed());

        let result = engine
            .query()
            .series(1)
            .execute()
            .expect("query should succeed");

        assert_eq!(result.series_id, 1);
        assert_eq!(result.points.len(), 100);
        assert_eq!(result.chunks_scanned, 1);

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn test_columnar_backed_engine_time_range_scans_only_overlapping_chunks() {
        let (store, path) = temp_columnar_store("time_range");
        // Truncate to millisecond precision: chunk compression stores
        // timestamps at millisecond granularity, so a sub-millisecond
        // `Utc::now()` boundary would otherwise round away from the exact
        // query start and make the range filter's `>=` comparison flaky.
        let start = DateTime::from_timestamp_millis(Utc::now().timestamp_millis())
            .expect("valid timestamp");

        // Two disjoint chunks for the same series.
        let chunk1_points: Vec<DataPoint> = (0..50)
            .map(|i| DataPoint {
                timestamp: start + Duration::seconds(i),
                value: i as f64,
            })
            .collect();
        let chunk1 = TimeChunk::new(1, start, Duration::seconds(50), chunk1_points)
            .expect("chunk construction should succeed");

        let chunk2_start = start + Duration::hours(3);
        let chunk2_points: Vec<DataPoint> = (0..50)
            .map(|i| DataPoint {
                timestamp: chunk2_start + Duration::seconds(i),
                value: (100 + i) as f64,
            })
            .collect();
        let chunk2 = TimeChunk::new(1, chunk2_start, Duration::seconds(50), chunk2_points)
            .expect("chunk construction should succeed");

        store
            .write_chunk(&chunk1)
            .expect("chunk write should succeed");
        store
            .write_chunk(&chunk2)
            .expect("chunk write should succeed");

        let engine = QueryEngine::from_columnar_store(Arc::clone(&store));

        // Querying only the first chunk's range must not scan the second.
        let result = engine
            .query()
            .series(1)
            .time_range(start, start + Duration::seconds(50))
            .execute()
            .expect("query should succeed");

        assert_eq!(result.chunks_scanned, 1);
        assert_eq!(result.points.len(), 50);
        assert!(result.points.iter().all(|p| p.value < 100.0));

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn test_columnar_backed_engine_aggregation() {
        let (store, path) = temp_columnar_store("aggregation");
        let start = Utc::now();

        let points: Vec<DataPoint> = (0..100)
            .map(|i| DataPoint {
                timestamp: start + Duration::seconds(i),
                value: i as f64,
            })
            .collect();
        let chunk = TimeChunk::new(1, start, Duration::hours(2), points)
            .expect("chunk construction should succeed");
        store
            .write_chunk(&chunk)
            .expect("chunk write should succeed");

        let engine = QueryEngine::from_columnar_store(Arc::clone(&store));
        let result = engine
            .query()
            .series(1)
            .aggregate(Aggregation::Avg)
            .execute()
            .expect("query should succeed");

        let avg = result
            .aggregated_value
            .expect("aggregation should be present");
        assert!((avg - 49.5).abs() < 0.1);

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn test_add_chunk_is_noop_on_columnar_backed_engine() {
        let (store, path) = temp_columnar_store("add_chunk_noop");
        let mut engine = QueryEngine::from_columnar_store(Arc::clone(&store));

        let start = Utc::now();
        let points = vec![DataPoint {
            timestamp: start,
            value: 1.0,
        }];
        let chunk = TimeChunk::new(1, start, Duration::hours(2), points)
            .expect("chunk construction should succeed");
        // add_chunk() must not panic and must not make the point queryable,
        // since it was never written through ColumnarStore::write_chunk.
        engine.add_chunk(chunk);

        let result = engine
            .query()
            .series(1)
            .execute()
            .expect("query should succeed");
        assert!(result.points.is_empty());

        let _ = std::fs::remove_dir_all(&path);
    }
}
