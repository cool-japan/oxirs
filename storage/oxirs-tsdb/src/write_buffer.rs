//! In-memory write buffer for time-series data ingestion.
//!
//! Accumulates incoming data points before flushing them to durable storage.
//! Supports size-based, time-based, and explicit flush policies, WAL
//! integration hooks, out-of-order write handling, backpressure, and partial
//! flushing.
//!
//! # Example
//!
//! ```
//! use oxirs_tsdb::write_buffer::{WriteBufferConfig, WriteBuffer, FlushPolicy, DataPoint};
//!
//! let config = WriteBufferConfig {
//!     max_capacity: 1000,
//!     flush_policy: FlushPolicy::SizeBased { threshold: 500 },
//!     ..Default::default()
//! };
//! let mut buf = WriteBuffer::new(config);
//! buf.push(DataPoint { series_id: 1, timestamp_ms: 1_000, value: 3.14 }).expect("push failed");
//! if buf.should_flush() {
//!     let points = buf.flush().expect("flush failed");
//!     println!("flushed {} points", points.len());
//! }
//! ```

use std::cmp::Ordering;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the write buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteBufferError {
    /// The buffer is full and backpressure is active.
    BufferFull,
    /// An explicit flush was requested while the buffer is in a transient state.
    FlushConflict(String),
    /// The WAL entry could not be written (simulated).
    WalError(String),
    /// General internal error.
    Internal(String),
}

impl std::fmt::Display for WriteBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferFull => write!(f, "Write buffer full: backpressure active"),
            Self::FlushConflict(msg) => write!(f, "Flush conflict: {msg}"),
            Self::WalError(msg) => write!(f, "WAL error: {msg}"),
            Self::Internal(msg) => write!(f, "Internal error: {msg}"),
        }
    }
}

impl std::error::Error for WriteBufferError {}

/// Result alias for write buffer operations.
pub type WriteBufferResult<T> = Result<T, WriteBufferError>;

// ---------------------------------------------------------------------------
// Data point
// ---------------------------------------------------------------------------

/// A single time-series observation.
#[derive(Debug, Clone, PartialEq)]
pub struct DataPoint {
    /// Numeric series identifier.
    pub series_id: u64,
    /// Millisecond UTC timestamp.
    pub timestamp_ms: i64,
    /// Observed value.
    pub value: f64,
}

impl Eq for DataPoint {}

impl PartialOrd for DataPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DataPoint {
    fn cmp(&self, other: &Self) -> Ordering {
        self.timestamp_ms
            .cmp(&other.timestamp_ms)
            .then(self.series_id.cmp(&other.series_id))
    }
}

// ---------------------------------------------------------------------------
// Flush policy
// ---------------------------------------------------------------------------

/// Determines when the buffer should be automatically flushed.
#[derive(Debug, Clone, PartialEq)]
pub enum FlushPolicy {
    /// Flush only when explicitly requested.
    Explicit,
    /// Flush when the number of buffered points reaches `threshold`.
    SizeBased {
        /// Number of data points that triggers a flush.
        threshold: usize,
    },
    /// Flush when the oldest buffered point exceeds `max_age`.
    TimeBased {
        /// Maximum age of the oldest buffered point before flushing.
        max_age: Duration,
    },
    /// Flush when either the size or time condition is met.
    Combined {
        /// Point count threshold.
        threshold: usize,
        /// Age threshold.
        max_age: Duration,
    },
}

impl Default for FlushPolicy {
    fn default() -> Self {
        Self::SizeBased { threshold: 1000 }
    }
}

// ---------------------------------------------------------------------------
// Buffer state
// ---------------------------------------------------------------------------

/// Lifecycle state of the write buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferState {
    /// No data points are buffered.
    Empty,
    /// Points are being accumulated normally.
    Accumulating,
    /// A flush is currently in progress.
    Flushing,
    /// The buffer is at maximum capacity; writes are blocked.
    Full,
}

// ---------------------------------------------------------------------------
// WAL entry
// ---------------------------------------------------------------------------

/// A simulated write-ahead log entry produced during a flush.
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// Sequence number assigned to this flush batch.
    pub sequence: u64,
    /// Number of data points in this entry.
    pub point_count: usize,
    /// Minimum timestamp in the batch (ms).
    pub min_timestamp_ms: i64,
    /// Maximum timestamp in the batch (ms).
    pub max_timestamp_ms: i64,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the write buffer.
#[derive(Debug, Clone)]
pub struct WriteBufferConfig {
    /// Maximum number of data points the buffer can hold before backpressure.
    pub max_capacity: usize,
    /// Flush policy.
    pub flush_policy: FlushPolicy,
    /// If true, WAL entries are produced on each flush.
    pub enable_wal: bool,
    /// Maximum age of buffered data before a partial flush is triggered (0 = disabled).
    pub partial_flush_age: Duration,
    /// Number of oldest points to flush in a partial flush (0 = all eligible).
    pub partial_flush_count: usize,
}

impl Default for WriteBufferConfig {
    fn default() -> Self {
        Self {
            max_capacity: 100_000,
            flush_policy: FlushPolicy::default(),
            enable_wal: true,
            partial_flush_age: Duration::ZERO,
            partial_flush_count: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Buffer statistics
// ---------------------------------------------------------------------------

/// Statistics about the current state and historical activity of the buffer.
#[derive(Debug, Clone, Default)]
pub struct BufferStats {
    /// Number of data points currently in the buffer.
    pub buffered_count: usize,
    /// Oldest timestamp (ms) in the buffer (`i64::MAX` if empty).
    pub oldest_timestamp_ms: i64,
    /// Newest timestamp (ms) in the buffer (`i64::MIN` if empty).
    pub newest_timestamp_ms: i64,
    /// Total number of flushes performed.
    pub flush_count: u64,
    /// Total number of data points flushed across all flushes.
    pub total_points_flushed: u64,
    /// Total number of out-of-order points that were sorted on flush.
    pub out_of_order_count: u64,
    /// Number of writes blocked by backpressure.
    pub backpressure_events: u64,
    /// Number of WAL entries written.
    pub wal_entries: u64,
}

// ---------------------------------------------------------------------------
// WAL sink (callback)
// ---------------------------------------------------------------------------

/// Trait for sinking WAL entries to durable storage.
///
/// Implement this on a real WAL to integrate durability.
pub trait WalSink: Send + 'static {
    /// Write a WAL entry.
    fn write_entry(&mut self, entry: WalEntry) -> WriteBufferResult<()>;
}

/// A no-op WAL sink (drops all entries).
pub struct NoopWalSink;

impl WalSink for NoopWalSink {
    fn write_entry(&mut self, _entry: WalEntry) -> WriteBufferResult<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Write buffer
// ---------------------------------------------------------------------------

/// In-memory write buffer for time-series data points.
pub struct WriteBuffer {
    config: WriteBufferConfig,
    points: Vec<DataPoint>,
    state: BufferState,
    stats: BufferStats,
    wal_sequence: u64,
    oldest_insert_time: Option<Instant>,
}

impl WriteBuffer {
    /// Create a new write buffer with the given configuration.
    pub fn new(config: WriteBufferConfig) -> Self {
        Self {
            config,
            points: Vec::new(),
            state: BufferState::Empty,
            stats: BufferStats {
                oldest_timestamp_ms: i64::MAX,
                newest_timestamp_ms: i64::MIN,
                ..Default::default()
            },
            wal_sequence: 0,
            oldest_insert_time: None,
        }
    }

    // ------------------------------------------------------------------
    // State queries
    // ------------------------------------------------------------------

    /// Current buffer state.
    pub fn state(&self) -> BufferState {
        self.state
    }

    /// Return a snapshot of current buffer statistics.
    pub fn stats(&self) -> &BufferStats {
        &self.stats
    }

    /// Number of data points currently held.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// True if no points are currently buffered.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// True if the buffer has reached maximum capacity.
    pub fn is_full(&self) -> bool {
        self.points.len() >= self.config.max_capacity
    }

    // ------------------------------------------------------------------
    // Push
    // ------------------------------------------------------------------

    /// Push a data point into the buffer.
    ///
    /// # Errors
    ///
    /// Returns [`WriteBufferError::BufferFull`] when the buffer is at capacity
    /// and backpressure should be applied by the caller.
    pub fn push(&mut self, point: DataPoint) -> WriteBufferResult<()> {
        if self.points.len() >= self.config.max_capacity {
            self.stats.backpressure_events += 1;
            self.state = BufferState::Full;
            return Err(WriteBufferError::BufferFull);
        }

        // Update timestamp bounds.
        if point.timestamp_ms < self.stats.oldest_timestamp_ms {
            self.stats.oldest_timestamp_ms = point.timestamp_ms;
        }
        if point.timestamp_ms > self.stats.newest_timestamp_ms {
            self.stats.newest_timestamp_ms = point.timestamp_ms;
        }

        self.points.push(point);
        self.stats.buffered_count = self.points.len();

        if self.oldest_insert_time.is_none() {
            self.oldest_insert_time = Some(Instant::now());
        }

        self.state = if self.points.len() >= self.config.max_capacity {
            BufferState::Full
        } else {
            BufferState::Accumulating
        };

        Ok(())
    }

    /// Push multiple data points at once.
    ///
    /// Points are added until the buffer is full; returns the number actually
    /// pushed.
    pub fn push_batch(&mut self, points: impl IntoIterator<Item = DataPoint>) -> usize {
        let mut count = 0usize;
        for point in points {
            if self.push(point).is_err() {
                break;
            }
            count += 1;
        }
        count
    }

    // ------------------------------------------------------------------
    // Flush triggering
    // ------------------------------------------------------------------

    /// True if the current policy conditions require a flush.
    pub fn should_flush(&self) -> bool {
        if self.points.is_empty() {
            return false;
        }
        match &self.config.flush_policy {
            FlushPolicy::Explicit => false,
            FlushPolicy::SizeBased { threshold } => self.points.len() >= *threshold,
            FlushPolicy::TimeBased { max_age } => self
                .oldest_insert_time
                .map(|t| t.elapsed() >= *max_age)
                .unwrap_or(false),
            FlushPolicy::Combined { threshold, max_age } => {
                let size_ok = self.points.len() >= *threshold;
                let time_ok = self
                    .oldest_insert_time
                    .map(|t| t.elapsed() >= *max_age)
                    .unwrap_or(false);
                size_ok || time_ok
            }
        }
    }

    // ------------------------------------------------------------------
    // Flush (full)
    // ------------------------------------------------------------------

    /// Flush all buffered data points, sorted by timestamp.
    ///
    /// Writes a WAL entry if `enable_wal` is set and a sink is provided.
    /// Returns the flushed points in ascending timestamp order.
    pub fn flush(&mut self) -> WriteBufferResult<Vec<DataPoint>> {
        self.flush_inner(None)
    }

    /// Flush with a WAL sink.
    pub fn flush_with_wal<W: WalSink>(&mut self, wal: &mut W) -> WriteBufferResult<Vec<DataPoint>> {
        self.flush_inner(Some(wal as &mut dyn WalSink))
    }

    fn flush_inner(&mut self, wal: Option<&mut dyn WalSink>) -> WriteBufferResult<Vec<DataPoint>> {
        if matches!(self.state, BufferState::Flushing) {
            return Err(WriteBufferError::FlushConflict(
                "flush already in progress".to_string(),
            ));
        }

        let prev_state = self.state;
        self.state = BufferState::Flushing;

        let mut points = std::mem::take(&mut self.points);

        // Detect out-of-order writes.
        let was_sorted = points
            .windows(2)
            .all(|w| w[0].timestamp_ms <= w[1].timestamp_ms);
        if !was_sorted {
            self.stats.out_of_order_count += points.len() as u64;
            points.sort_unstable();
        }

        // WAL integration.
        if self.config.enable_wal {
            let entry = self.build_wal_entry(&points);
            if let Some(sink) = wal {
                sink.write_entry(entry)?;
            }
            self.stats.wal_entries += 1;
        }

        // Update statistics.
        self.stats.flush_count += 1;
        self.stats.total_points_flushed += points.len() as u64;
        self.stats.buffered_count = 0;
        self.stats.oldest_timestamp_ms = i64::MAX;
        self.stats.newest_timestamp_ms = i64::MIN;
        self.oldest_insert_time = None;

        let _ = prev_state;
        self.state = BufferState::Empty;

        Ok(points)
    }

    // ------------------------------------------------------------------
    // Partial flush
    // ------------------------------------------------------------------

    /// Flush only the oldest data points (those with the smallest timestamps).
    ///
    /// `count` specifies how many points to flush (0 = use config default).
    /// Points are sorted by timestamp before selection.
    pub fn partial_flush(&mut self, count: usize) -> WriteBufferResult<Vec<DataPoint>> {
        if matches!(self.state, BufferState::Flushing) {
            return Err(WriteBufferError::FlushConflict(
                "flush already in progress".to_string(),
            ));
        }
        if self.points.is_empty() {
            return Ok(Vec::new());
        }

        self.state = BufferState::Flushing;

        // Sort all points so we can take the oldest.
        self.points.sort_unstable();

        let take = if count == 0 {
            self.config.partial_flush_count.max(1)
        } else {
            count
        };
        let take = take.min(self.points.len());

        let flushed: Vec<DataPoint> = self.points.drain(..take).collect();

        // Update stats.
        self.stats.flush_count += 1;
        self.stats.total_points_flushed += flushed.len() as u64;
        self.stats.buffered_count = self.points.len();

        // Recompute timestamp bounds.
        self.recompute_bounds();

        self.state = if self.points.is_empty() {
            BufferState::Empty
        } else {
            BufferState::Accumulating
        };

        Ok(flushed)
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn build_wal_entry(&mut self, points: &[DataPoint]) -> WalEntry {
        self.wal_sequence += 1;
        let min_ts = points.iter().map(|p| p.timestamp_ms).min().unwrap_or(0);
        let max_ts = points.iter().map(|p| p.timestamp_ms).max().unwrap_or(0);
        WalEntry {
            sequence: self.wal_sequence,
            point_count: points.len(),
            min_timestamp_ms: min_ts,
            max_timestamp_ms: max_ts,
        }
    }

    fn recompute_bounds(&mut self) {
        if self.points.is_empty() {
            self.stats.oldest_timestamp_ms = i64::MAX;
            self.stats.newest_timestamp_ms = i64::MIN;
        } else {
            self.stats.oldest_timestamp_ms = self
                .points
                .iter()
                .map(|p| p.timestamp_ms)
                .min()
                .unwrap_or(i64::MAX);
            self.stats.newest_timestamp_ms = self
                .points
                .iter()
                .map(|p| p.timestamp_ms)
                .max()
                .unwrap_or(i64::MIN);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_point(series_id: u64, timestamp_ms: i64, value: f64) -> DataPoint {
        DataPoint {
            series_id,
            timestamp_ms,
            value,
        }
    }

    fn size_buffer(threshold: usize) -> WriteBuffer {
        WriteBuffer::new(WriteBufferConfig {
            max_capacity: 10_000,
            flush_policy: FlushPolicy::SizeBased { threshold },
            enable_wal: false,
            ..Default::default()
        })
    }

    // -- state transitions --------------------------------------------------

    #[test]
    fn test_initial_state_is_empty() {
        let buf = WriteBuffer::new(WriteBufferConfig::default());
        assert_eq!(buf.state(), BufferState::Empty);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_push_transitions_to_accumulating() {
        let mut buf = size_buffer(100);
        buf.push(make_point(1, 1000, 1.0)).expect("push failed");
        assert_eq!(buf.state(), BufferState::Accumulating);
    }

    #[test]
    fn test_flush_transitions_back_to_empty() {
        let mut buf = size_buffer(100);
        buf.push(make_point(1, 1000, 1.0)).expect("push failed");
        let _ = buf.flush().expect("flush failed");
        assert_eq!(buf.state(), BufferState::Empty);
    }

    #[test]
    fn test_buffer_full_state_on_capacity() {
        let config = WriteBufferConfig {
            max_capacity: 3,
            flush_policy: FlushPolicy::Explicit,
            enable_wal: false,
            ..Default::default()
        };
        let mut buf = WriteBuffer::new(config);
        buf.push(make_point(1, 1, 1.0)).expect("push 1");
        buf.push(make_point(1, 2, 2.0)).expect("push 2");
        buf.push(make_point(1, 3, 3.0)).expect("push 3");
        assert_eq!(buf.state(), BufferState::Full);
    }

    // -- backpressure -------------------------------------------------------

    #[test]
    fn test_push_beyond_capacity_returns_buffer_full() {
        let config = WriteBufferConfig {
            max_capacity: 2,
            flush_policy: FlushPolicy::Explicit,
            enable_wal: false,
            ..Default::default()
        };
        let mut buf = WriteBuffer::new(config);
        buf.push(make_point(1, 1, 0.0)).expect("push 1");
        buf.push(make_point(1, 2, 0.0)).expect("push 2");
        let err = buf.push(make_point(1, 3, 0.0)).unwrap_err();
        assert_eq!(err, WriteBufferError::BufferFull);
    }

    #[test]
    fn test_backpressure_events_counted() {
        let config = WriteBufferConfig {
            max_capacity: 1,
            flush_policy: FlushPolicy::Explicit,
            enable_wal: false,
            ..Default::default()
        };
        let mut buf = WriteBuffer::new(config);
        buf.push(make_point(1, 1, 0.0)).expect("push 1");
        let _ = buf.push(make_point(1, 2, 0.0));
        let _ = buf.push(make_point(1, 3, 0.0));
        assert_eq!(buf.stats().backpressure_events, 2);
    }

    // -- push_batch ---------------------------------------------------------

    #[test]
    fn test_push_batch_stops_at_capacity() {
        let config = WriteBufferConfig {
            max_capacity: 3,
            flush_policy: FlushPolicy::Explicit,
            enable_wal: false,
            ..Default::default()
        };
        let mut buf = WriteBuffer::new(config);
        let points: Vec<DataPoint> = (0..10).map(|i| make_point(1, i, 0.0)).collect();
        let pushed = buf.push_batch(points);
        assert_eq!(pushed, 3);
    }

    // -- flush ordering -----------------------------------------------------

    #[test]
    fn test_flush_returns_points_sorted_by_timestamp() {
        let mut buf = size_buffer(10);
        buf.push(make_point(1, 300, 1.0)).expect("push");
        buf.push(make_point(1, 100, 2.0)).expect("push");
        buf.push(make_point(1, 200, 3.0)).expect("push");
        let flushed = buf.flush().expect("flush failed");
        assert_eq!(flushed[0].timestamp_ms, 100);
        assert_eq!(flushed[1].timestamp_ms, 200);
        assert_eq!(flushed[2].timestamp_ms, 300);
    }

    #[test]
    fn test_out_of_order_counted_on_flush() {
        let mut buf = size_buffer(10);
        buf.push(make_point(1, 200, 1.0)).expect("push");
        buf.push(make_point(1, 100, 2.0)).expect("push");
        let _ = buf.flush().expect("flush");
        assert_eq!(buf.stats().out_of_order_count, 2);
    }

    #[test]
    fn test_in_order_write_not_counted_as_out_of_order() {
        let mut buf = size_buffer(10);
        buf.push(make_point(1, 100, 1.0)).expect("push");
        buf.push(make_point(1, 200, 2.0)).expect("push");
        let _ = buf.flush().expect("flush");
        assert_eq!(buf.stats().out_of_order_count, 0);
    }

    // -- flush stats --------------------------------------------------------

    #[test]
    fn test_flush_count_increments() {
        let mut buf = size_buffer(10);
        buf.push(make_point(1, 1, 0.0)).expect("push");
        buf.flush().expect("flush 1");
        buf.push(make_point(1, 2, 0.0)).expect("push");
        buf.flush().expect("flush 2");
        assert_eq!(buf.stats().flush_count, 2);
    }

    #[test]
    fn test_total_points_flushed_accumulates() {
        let mut buf = size_buffer(10);
        for i in 0..5 {
            buf.push(make_point(1, i, 0.0)).expect("push");
        }
        buf.flush().expect("flush 1");
        for i in 5..8 {
            buf.push(make_point(1, i, 0.0)).expect("push");
        }
        buf.flush().expect("flush 2");
        assert_eq!(buf.stats().total_points_flushed, 8);
    }

    #[test]
    fn test_buffered_count_reset_after_flush() {
        let mut buf = size_buffer(10);
        buf.push(make_point(1, 1, 0.0)).expect("push");
        buf.flush().expect("flush");
        assert_eq!(buf.stats().buffered_count, 0);
    }

    // -- timestamp bounds ---------------------------------------------------

    #[test]
    fn test_timestamp_bounds_updated_on_push() {
        let mut buf = size_buffer(10);
        buf.push(make_point(1, 500, 0.0)).expect("push");
        buf.push(make_point(1, 100, 0.0)).expect("push");
        buf.push(make_point(1, 900, 0.0)).expect("push");
        assert_eq!(buf.stats().oldest_timestamp_ms, 100);
        assert_eq!(buf.stats().newest_timestamp_ms, 900);
    }

    #[test]
    fn test_timestamp_bounds_reset_after_flush() {
        let mut buf = size_buffer(10);
        buf.push(make_point(1, 100, 0.0)).expect("push");
        buf.flush().expect("flush");
        assert_eq!(buf.stats().oldest_timestamp_ms, i64::MAX);
        assert_eq!(buf.stats().newest_timestamp_ms, i64::MIN);
    }

    // -- flush policy -------------------------------------------------------

    #[test]
    fn test_should_flush_size_based_below_threshold() {
        let mut buf = size_buffer(5);
        buf.push(make_point(1, 1, 0.0)).expect("push");
        buf.push(make_point(1, 2, 0.0)).expect("push");
        assert!(!buf.should_flush());
    }

    #[test]
    fn test_should_flush_size_based_at_threshold() {
        let mut buf = size_buffer(2);
        buf.push(make_point(1, 1, 0.0)).expect("push");
        buf.push(make_point(1, 2, 0.0)).expect("push");
        assert!(buf.should_flush());
    }

    #[test]
    fn test_should_flush_explicit_never_auto() {
        let config = WriteBufferConfig {
            max_capacity: 10_000,
            flush_policy: FlushPolicy::Explicit,
            enable_wal: false,
            ..Default::default()
        };
        let mut buf = WriteBuffer::new(config);
        for i in 0..100 {
            buf.push(make_point(1, i, 0.0)).expect("push");
        }
        assert!(!buf.should_flush());
    }

    #[test]
    fn test_should_flush_empty_buffer_false() {
        let buf = size_buffer(1);
        assert!(!buf.should_flush());
    }

    #[test]
    fn test_should_flush_combined_size_path() {
        let config = WriteBufferConfig {
            max_capacity: 10_000,
            flush_policy: FlushPolicy::Combined {
                threshold: 2,
                max_age: Duration::from_secs(3600),
            },
            enable_wal: false,
            ..Default::default()
        };
        let mut buf = WriteBuffer::new(config);
        buf.push(make_point(1, 1, 0.0)).expect("push");
        buf.push(make_point(1, 2, 0.0)).expect("push");
        assert!(buf.should_flush());
    }

    // -- partial flush ------------------------------------------------------

    #[test]
    fn test_partial_flush_removes_oldest_points() {
        let mut buf = size_buffer(100);
        buf.push(make_point(1, 300, 1.0)).expect("push");
        buf.push(make_point(1, 100, 2.0)).expect("push");
        buf.push(make_point(1, 200, 3.0)).expect("push");
        let flushed = buf.partial_flush(1).expect("partial flush");
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].timestamp_ms, 100);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_partial_flush_empty_buffer_returns_empty() {
        let mut buf = size_buffer(100);
        let flushed = buf.partial_flush(5).expect("partial flush");
        assert!(flushed.is_empty());
    }

    #[test]
    fn test_partial_flush_count_exceeds_buffer_size() {
        let mut buf = size_buffer(100);
        buf.push(make_point(1, 1, 0.0)).expect("push");
        buf.push(make_point(1, 2, 0.0)).expect("push");
        let flushed = buf.partial_flush(10).expect("partial flush");
        assert_eq!(flushed.len(), 2);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_partial_flush_state_after_partial() {
        let mut buf = size_buffer(100);
        buf.push(make_point(1, 1, 0.0)).expect("push");
        buf.push(make_point(1, 2, 0.0)).expect("push");
        let _ = buf.partial_flush(1).expect("partial flush");
        assert_eq!(buf.state(), BufferState::Accumulating);
    }

    // -- WAL integration ----------------------------------------------------

    #[test]
    fn test_wal_entry_count_increments_on_flush() {
        let config = WriteBufferConfig {
            max_capacity: 10_000,
            flush_policy: FlushPolicy::Explicit,
            enable_wal: true,
            ..Default::default()
        };
        let mut buf = WriteBuffer::new(config);
        buf.push(make_point(1, 100, 1.0)).expect("push");
        buf.flush().expect("flush");
        assert_eq!(buf.stats().wal_entries, 1);
    }

    #[test]
    fn test_wal_disabled_no_entries_recorded() {
        let config = WriteBufferConfig {
            max_capacity: 10_000,
            flush_policy: FlushPolicy::Explicit,
            enable_wal: false,
            ..Default::default()
        };
        let mut buf = WriteBuffer::new(config);
        buf.push(make_point(1, 100, 1.0)).expect("push");
        buf.flush().expect("flush");
        assert_eq!(buf.stats().wal_entries, 0);
    }

    // -- error types --------------------------------------------------------

    #[test]
    fn test_write_buffer_error_display() {
        assert!(WriteBufferError::BufferFull
            .to_string()
            .contains("backpressure"));
        assert!(WriteBufferError::FlushConflict("x".into())
            .to_string()
            .contains("x"));
        assert!(WriteBufferError::WalError("wal".into())
            .to_string()
            .contains("wal"));
        assert!(WriteBufferError::Internal("internal".into())
            .to_string()
            .contains("internal"));
    }

    // -- len / is_empty / is_full -------------------------------------------

    #[test]
    fn test_len_reflects_buffered_points() {
        let mut buf = size_buffer(100);
        for i in 0..7 {
            buf.push(make_point(1, i, 0.0)).expect("push");
        }
        assert_eq!(buf.len(), 7);
    }

    #[test]
    fn test_is_full_at_capacity() {
        let config = WriteBufferConfig {
            max_capacity: 2,
            flush_policy: FlushPolicy::Explicit,
            enable_wal: false,
            ..Default::default()
        };
        let mut buf = WriteBuffer::new(config);
        buf.push(make_point(1, 1, 0.0)).expect("push");
        assert!(!buf.is_full());
        buf.push(make_point(1, 2, 0.0)).expect("push");
        assert!(buf.is_full());
    }

    // -- DataPoint ordering -------------------------------------------------

    #[test]
    fn test_data_point_ordering_by_timestamp() {
        let a = make_point(1, 100, 0.0);
        let b = make_point(1, 200, 0.0);
        assert!(a < b);
    }

    #[test]
    fn test_data_point_ordering_same_timestamp_by_series() {
        let a = make_point(1, 100, 0.0);
        let b = make_point(2, 100, 0.0);
        assert!(a < b);
    }

    // -- multiple series in one flush ---------------------------------------

    #[test]
    fn test_flush_multiple_series_interleaved() {
        let mut buf = size_buffer(100);
        buf.push(make_point(2, 200, 1.0)).expect("push");
        buf.push(make_point(1, 100, 2.0)).expect("push");
        buf.push(make_point(3, 150, 3.0)).expect("push");
        let flushed = buf.flush().expect("flush");
        assert_eq!(flushed[0].series_id, 1);
        assert_eq!(flushed[1].series_id, 3);
        assert_eq!(flushed[2].series_id, 2);
    }

    // -- flush_with_wal -----------------------------------------------------

    #[test]
    fn test_flush_with_noop_wal_sink() {
        let mut buf = WriteBuffer::new(WriteBufferConfig {
            max_capacity: 10_000,
            flush_policy: FlushPolicy::Explicit,
            enable_wal: true,
            ..Default::default()
        });
        buf.push(make_point(1, 100, 1.0)).expect("push");
        let mut sink = NoopWalSink;
        let flushed = buf.flush_with_wal(&mut sink).expect("flush_with_wal");
        assert_eq!(flushed.len(), 1);
    }
}
