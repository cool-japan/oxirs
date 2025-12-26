//! Range query implementation
//!
//! Provides efficient time-range queries over time-series data.

use crate::error::TsdbResult;
use crate::series::DataPoint;
use crate::storage::TimeChunk;
use chrono::{DateTime, Utc};

/// Time range specification
#[derive(Debug, Clone, Copy)]
pub struct TimeRange {
    /// Start of time range (inclusive)
    pub start: DateTime<Utc>,
    /// End of time range (exclusive)
    pub end: DateTime<Utc>,
}

impl TimeRange {
    /// Create a new time range
    pub fn new(start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        Self { start, end }
    }

    /// Check if a timestamp falls within this range
    pub fn contains(&self, timestamp: DateTime<Utc>) -> bool {
        timestamp >= self.start && timestamp < self.end
    }

    /// Check if this range overlaps with a chunk's time range
    pub fn overlaps_chunk(&self, chunk: &TimeChunk) -> bool {
        // Ranges overlap if one doesn't end before the other starts
        self.start < chunk.end_time && self.end > chunk.start_time
    }

    /// Duration of the time range
    pub fn duration(&self) -> chrono::Duration {
        self.end - self.start
    }
}

/// Range query over time-series data
#[derive(Debug)]
pub struct RangeQuery {
    /// Series to query
    pub series_id: u64,
    /// Time range
    pub time_range: TimeRange,
    /// Maximum number of results (None = unlimited)
    pub limit: Option<usize>,
    /// Order by timestamp (true = ascending, false = descending)
    pub ascending: bool,
}

impl RangeQuery {
    /// Create a new range query
    pub fn new(series_id: u64, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        Self {
            series_id,
            time_range: TimeRange::new(start, end),
            limit: None,
            ascending: true,
        }
    }

    /// Set result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set order (ascending or descending)
    pub fn with_order(mut self, ascending: bool) -> Self {
        self.ascending = ascending;
        self
    }

    /// Execute query against a list of chunks
    pub fn execute(&self, chunks: &[TimeChunk]) -> TsdbResult<Vec<DataPoint>> {
        let mut results = Vec::new();

        // Filter chunks that overlap with time range
        let relevant_chunks: Vec<&TimeChunk> = chunks
            .iter()
            .filter(|c| c.series_id == self.series_id && self.time_range.overlaps_chunk(c))
            .collect();

        // Query each relevant chunk
        for chunk in relevant_chunks {
            let chunk_results = chunk.query_range(self.time_range.start, self.time_range.end)?;
            results.extend(chunk_results);

            // Early exit if limit reached
            if let Some(limit) = self.limit {
                if results.len() >= limit {
                    results.truncate(limit);
                    break;
                }
            }
        }

        // Sort by timestamp
        if self.ascending {
            results.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        } else {
            results.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        }

        // Apply limit after sorting
        if let Some(limit) = self.limit {
            results.truncate(limit);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn create_test_chunk(series_id: u64, start: DateTime<Utc>, count: usize) -> TimeChunk {
        let mut points = Vec::new();
        for i in 0..count {
            points.push(DataPoint {
                timestamp: start + Duration::seconds(i as i64),
                value: i as f64,
            });
        }
        TimeChunk::new(series_id, start, Duration::hours(2), points).unwrap()
    }

    #[test]
    fn test_time_range_contains() {
        let now = Utc::now();
        let range = TimeRange::new(now, now + Duration::hours(1));

        assert!(range.contains(now));
        assert!(range.contains(now + Duration::minutes(30)));
        assert!(!range.contains(now - Duration::minutes(1)));
        assert!(!range.contains(now + Duration::hours(1))); // End is exclusive
    }

    #[test]
    fn test_range_query_basic() {
        let now = Utc::now();
        let chunk = create_test_chunk(1, now, 100);

        let query = RangeQuery::new(1, now + Duration::seconds(10), now + Duration::seconds(20));
        let results = query.execute(&[chunk]).unwrap();

        assert_eq!(results.len(), 10);
        assert!(results[0].timestamp >= now + Duration::seconds(10));
    }

    #[test]
    fn test_range_query_with_limit() {
        let now = Utc::now();
        let chunk = create_test_chunk(1, now, 100);

        let query = RangeQuery::new(1, now, now + Duration::seconds(100)).with_limit(5);
        let results = query.execute(&[chunk]).unwrap();

        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_range_query_descending() {
        let now = Utc::now();
        let chunk = create_test_chunk(1, now, 100);

        let query = RangeQuery::new(1, now, now + Duration::seconds(100)).with_order(false);
        let results = query.execute(&[chunk]).unwrap();

        // Should be descending
        for window in results.windows(2) {
            assert!(window[0].timestamp >= window[1].timestamp);
        }
    }

    #[test]
    fn test_range_overlaps_chunk() {
        let now = Utc::now();
        let chunk = create_test_chunk(1, now, 100);

        // Range fully contains chunk
        let range1 = TimeRange::new(now - Duration::hours(1), now + Duration::hours(3));
        assert!(range1.overlaps_chunk(&chunk));

        // Range partially overlaps
        let range2 = TimeRange::new(now + Duration::seconds(50), now + Duration::hours(3));
        assert!(range2.overlaps_chunk(&chunk));

        // Range doesn't overlap (before chunk)
        let range3 = TimeRange::new(now - Duration::hours(2), now - Duration::hours(1));
        assert!(!range3.overlaps_chunk(&chunk));
    }
}
