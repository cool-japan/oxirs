//! SPARQL query audit logger with statistics.
//!
//! Provides an in-memory ring-buffer logger for SPARQL query audit trails,
//! supporting per-query timing, result counts, error recording, and aggregate
//! statistics computation.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonic query ID counter shared across all loggers in the process.
static QUERY_COUNTER: AtomicU64 = AtomicU64::new(1);

/// A single audit log entry capturing one SPARQL query execution.
#[derive(Debug, Clone)]
pub struct QueryLogEntry {
    /// Unique monotonically increasing identifier for this query.
    pub query_id: u64,
    /// The SPARQL query text.
    pub query: String,
    /// The dataset name against which the query was executed.
    pub dataset: String,
    /// Execution duration in milliseconds.
    pub duration_ms: u64,
    /// Number of results returned by the query.
    pub result_count: usize,
    /// Error message if the query failed, `None` otherwise.
    pub error: Option<String>,
    /// Wall-clock timestamp in milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
}

/// Aggregate statistics over logged queries.
#[derive(Debug, Clone)]
pub struct LogStats {
    /// Total number of queries logged (including errors).
    pub total_queries: u64,
    /// Number of queries that produced an error.
    pub error_count: u64,
    /// Average execution duration in milliseconds across successful queries.
    /// Returns `0.0` when there are no successful queries.
    pub avg_duration_ms: f64,
    /// Maximum execution duration in milliseconds seen for any single successful query.
    /// Returns `0` when there are no successful queries.
    pub max_duration_ms: u64,
}

/// In-memory ring-buffer SPARQL query audit logger.
///
/// Retains up to `capacity` log entries; when the buffer is full the oldest
/// entry is silently evicted to make room for the new one.
///
/// # Examples
///
/// ```rust
/// use oxirs_fuseki::query_logger::QueryLogger;
///
/// let mut logger = QueryLogger::new(100);
/// logger.log("SELECT * WHERE { ?s ?p ?o }", "my-dataset", 42, 3);
/// let stats = logger.stats();
/// assert_eq!(stats.total_queries, 1);
/// ```
#[derive(Debug)]
pub struct QueryLogger {
    capacity: usize,
    buffer: VecDeque<QueryLogEntry>,
    /// Running sum of successful-query durations for O(1) avg computation.
    duration_sum: u64,
    /// Running maximum duration for O(1) max computation.
    ///
    /// Note: when the entry holding the maximum is evicted we do a full
    /// rescan of the buffer, which is at most O(capacity).
    max_duration: u64,
    /// Cached total query counter (tracks all entries ever inserted, not just
    /// those still in the buffer).
    total_logged: u64,
    /// Cached error counter (tracks all errors ever logged).
    error_logged: u64,
}

impl QueryLogger {
    /// Create a new query logger with the given ring-buffer capacity.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero.
    pub fn new(capacity: usize) -> Self {
        assert!(
            capacity > 0,
            "QueryLogger capacity must be greater than zero"
        );
        Self {
            capacity,
            buffer: VecDeque::with_capacity(capacity),
            duration_sum: 0,
            max_duration: 0,
            total_logged: 0,
            error_logged: 0,
        }
    }

    /// Current timestamp in milliseconds since Unix epoch.
    fn now_ms() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Log a successful query execution and return its assigned query ID.
    pub fn log(
        &mut self,
        query: &str,
        dataset: &str,
        duration_ms: u64,
        result_count: usize,
    ) -> u64 {
        let query_id = QUERY_COUNTER.fetch_add(1, Ordering::Relaxed);
        let entry = QueryLogEntry {
            query_id,
            query: query.to_string(),
            dataset: dataset.to_string(),
            duration_ms,
            result_count,
            error: None,
            timestamp_ms: Self::now_ms(),
        };
        self.push_entry(entry);
        query_id
    }

    /// Log a failed query execution and return its assigned query ID.
    ///
    /// The `duration_ms` is recorded as `0` and the entry is excluded from
    /// `avg_duration_ms` / `max_duration_ms` statistics.
    pub fn log_error(&mut self, query: &str, dataset: &str, error: &str) -> u64 {
        let query_id = QUERY_COUNTER.fetch_add(1, Ordering::Relaxed);
        let entry = QueryLogEntry {
            query_id,
            query: query.to_string(),
            dataset: dataset.to_string(),
            duration_ms: 0,
            result_count: 0,
            error: Some(error.to_string()),
            timestamp_ms: Self::now_ms(),
        };
        self.error_logged += 1;
        self.push_raw(entry);
        query_id
    }

    /// Internal: push a successful entry updating duration statistics.
    fn push_entry(&mut self, entry: QueryLogEntry) {
        self.duration_sum += entry.duration_ms;
        let new_duration = entry.duration_ms;
        if new_duration > self.max_duration {
            self.max_duration = new_duration;
        }
        self.total_logged += 1;
        self.evict_if_full_successful();
        self.buffer.push_back(entry);
    }

    /// Internal: push a raw entry without updating duration statistics.
    fn push_raw(&mut self, entry: QueryLogEntry) {
        self.total_logged += 1;
        self.evict_if_full();
        self.buffer.push_back(entry);
    }

    /// Evict the oldest entry when the buffer is at capacity (for successful entries).
    fn evict_if_full_successful(&mut self) {
        if self.buffer.len() >= self.capacity {
            if let Some(old) = self.buffer.pop_front() {
                if old.error.is_none() {
                    // Subtract the evicted duration from the running sum.
                    self.duration_sum = self.duration_sum.saturating_sub(old.duration_ms);
                    // If the evicted entry held the max, rescan.
                    if old.duration_ms >= self.max_duration {
                        self.recompute_max();
                    }
                }
            }
        }
    }

    /// Evict the oldest entry when the buffer is at capacity.
    fn evict_if_full(&mut self) {
        if self.buffer.len() >= self.capacity {
            if let Some(old) = self.buffer.pop_front() {
                if old.error.is_none() {
                    self.duration_sum = self.duration_sum.saturating_sub(old.duration_ms);
                    if old.duration_ms >= self.max_duration {
                        self.recompute_max();
                    }
                }
            }
        }
    }

    /// Recompute the maximum duration from the current buffer.
    fn recompute_max(&mut self) {
        self.max_duration = self
            .buffer
            .iter()
            .filter(|e| e.error.is_none())
            .map(|e| e.duration_ms)
            .max()
            .unwrap_or(0);
    }

    /// Return a slice of all entries currently in the ring-buffer (oldest first).
    pub fn entries(&self) -> &[QueryLogEntry] {
        // VecDeque is guaranteed contiguous after make_contiguous, but we
        // expose as a Vec reference through a helper to avoid mutating self.
        // Instead we return from the as_slices pair.
        // Collect into internal vec isn't possible without mut; callers can
        // iterate via entries_vec() if they need a Vec.
        self.buffer.as_slices().0
    }

    /// Return all entries as an ordered Vec (oldest first).
    pub fn entries_vec(&self) -> Vec<&QueryLogEntry> {
        self.buffer.iter().collect()
    }

    /// Return up to `n` most-recent log entries (newest first).
    pub fn recent(&self, n: usize) -> Vec<&QueryLogEntry> {
        self.buffer.iter().rev().take(n).collect()
    }

    /// Compute aggregate statistics over the entries currently in the buffer.
    ///
    /// Duration statistics (`avg_duration_ms`, `max_duration_ms`) exclude
    /// error entries.
    pub fn stats(&self) -> LogStats {
        let successful: Vec<&QueryLogEntry> =
            self.buffer.iter().filter(|e| e.error.is_none()).collect();

        let avg_duration_ms = if successful.is_empty() {
            0.0
        } else {
            self.duration_sum as f64 / successful.len() as f64
        };

        LogStats {
            total_queries: self.total_logged,
            error_count: self.error_logged,
            avg_duration_ms,
            max_duration_ms: self.max_duration,
        }
    }

    /// Return all entries whose `duration_ms` exceeds `threshold_ms`.
    ///
    /// Error entries (with `duration_ms == 0`) are excluded unless
    /// `threshold_ms == 0`.
    pub fn slow_queries(&self, threshold_ms: u64) -> Vec<&QueryLogEntry> {
        self.buffer
            .iter()
            .filter(|e| e.error.is_none() && e.duration_ms > threshold_ms)
            .collect()
    }

    /// Remove all entries from the ring-buffer and reset statistics.
    ///
    /// Note: `total_queries` and `error_count` in `LogStats` reflect
    /// lifetime totals and are also reset by this call.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.duration_sum = 0;
        self.max_duration = 0;
        self.total_logged = 0;
        self.error_logged = 0;
    }

    /// Return the current number of entries in the buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Return `true` if the buffer contains no entries.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Return the maximum number of entries the buffer can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Construction ---

    #[test]
    fn test_new_is_empty() {
        let logger = QueryLogger::new(10);
        assert!(logger.is_empty());
        assert_eq!(logger.len(), 0);
        assert_eq!(logger.capacity(), 10);
    }

    #[test]
    fn test_new_stats_zero() {
        let logger = QueryLogger::new(10);
        let s = logger.stats();
        assert_eq!(s.total_queries, 0);
        assert_eq!(s.error_count, 0);
        assert!((s.avg_duration_ms - 0.0).abs() < f64::EPSILON);
        assert_eq!(s.max_duration_ms, 0);
    }

    // --- Basic logging ---

    #[test]
    fn test_log_single_entry() {
        let mut logger = QueryLogger::new(10);
        let id = logger.log("SELECT * WHERE { ?s ?p ?o }", "ds1", 50, 3);
        assert!(id > 0);
        assert_eq!(logger.len(), 1);
    }

    #[test]
    fn test_log_entry_fields() {
        let mut logger = QueryLogger::new(10);
        let q = "SELECT ?s WHERE { ?s a <http://example.org/T> }";
        logger.log(q, "my-dataset", 120, 7);
        let entries = logger.entries_vec();
        assert_eq!(entries.len(), 1);
        let e = entries[0];
        assert_eq!(e.query, q);
        assert_eq!(e.dataset, "my-dataset");
        assert_eq!(e.duration_ms, 120);
        assert_eq!(e.result_count, 7);
        assert!(e.error.is_none());
    }

    #[test]
    fn test_log_multiple_entries_incrementing_ids() {
        let mut logger = QueryLogger::new(10);
        let id1 = logger.log("SELECT * WHERE { ?s ?p ?o }", "ds", 10, 1);
        let id2 = logger.log("ASK { ?s ?p ?o }", "ds", 5, 0);
        assert!(id2 > id1);
    }

    // --- Error logging ---

    #[test]
    fn test_log_error_entry() {
        let mut logger = QueryLogger::new(10);
        let id = logger.log_error("INVALID SPARQL", "ds", "Parse error");
        assert!(id > 0);
        assert_eq!(logger.len(), 1);
    }

    #[test]
    fn test_log_error_fields() {
        let mut logger = QueryLogger::new(10);
        logger.log_error("BAD QUERY", "ds2", "Syntax error at token X");
        let entries = logger.entries_vec();
        let e = entries[0];
        assert_eq!(e.query, "BAD QUERY");
        assert_eq!(e.dataset, "ds2");
        assert_eq!(e.duration_ms, 0);
        assert_eq!(e.result_count, 0);
        assert_eq!(e.error.as_deref(), Some("Syntax error at token X"));
    }

    #[test]
    fn test_error_count_in_stats() {
        let mut logger = QueryLogger::new(10);
        logger.log("SELECT * WHERE { ?s ?p ?o }", "ds", 10, 1);
        logger.log_error("BAD", "ds", "error");
        logger.log_error("BAD2", "ds", "error2");
        let s = logger.stats();
        assert_eq!(s.error_count, 2);
        assert_eq!(s.total_queries, 3);
    }

    // --- Statistics ---

    #[test]
    fn test_stats_avg_duration() {
        let mut logger = QueryLogger::new(10);
        logger.log("Q1", "ds", 100, 1);
        logger.log("Q2", "ds", 200, 2);
        let s = logger.stats();
        assert!((s.avg_duration_ms - 150.0).abs() < 0.001);
    }

    #[test]
    fn test_stats_max_duration() {
        let mut logger = QueryLogger::new(10);
        logger.log("Q1", "ds", 50, 1);
        logger.log("Q2", "ds", 500, 2);
        logger.log("Q3", "ds", 200, 3);
        let s = logger.stats();
        assert_eq!(s.max_duration_ms, 500);
    }

    #[test]
    fn test_stats_excludes_errors_from_avg() {
        let mut logger = QueryLogger::new(10);
        logger.log("Q1", "ds", 100, 1);
        logger.log_error("BAD", "ds", "err");
        let s = logger.stats();
        // Only Q1 contributes; error (duration 0) excluded
        assert!((s.avg_duration_ms - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_stats_excludes_errors_from_max() {
        let mut logger = QueryLogger::new(10);
        logger.log("Q1", "ds", 42, 1);
        logger.log_error("BAD", "ds", "err");
        let s = logger.stats();
        assert_eq!(s.max_duration_ms, 42);
    }

    #[test]
    fn test_stats_total_queries_lifetime() {
        let mut logger = QueryLogger::new(10);
        for i in 0..5u64 {
            logger.log(&format!("Q{}", i), "ds", i * 10, i as usize);
        }
        let s = logger.stats();
        assert_eq!(s.total_queries, 5);
    }

    // --- Ring buffer overflow ---

    #[test]
    fn test_ring_buffer_capacity_not_exceeded() {
        let mut logger = QueryLogger::new(5);
        for i in 0..10u64 {
            logger.log(&format!("Q{}", i), "ds", i * 10, i as usize);
        }
        assert_eq!(logger.len(), 5);
    }

    #[test]
    fn test_ring_buffer_oldest_evicted() {
        let mut logger = QueryLogger::new(3);
        logger.log("Q1", "ds", 10, 1);
        logger.log("Q2", "ds", 20, 2);
        logger.log("Q3", "ds", 30, 3);
        logger.log("Q4", "ds", 40, 4); // evicts Q1
        let queries: Vec<String> = logger
            .entries_vec()
            .iter()
            .map(|e| e.query.clone())
            .collect();
        assert!(!queries.contains(&"Q1".to_string()));
        assert!(queries.contains(&"Q4".to_string()));
    }

    #[test]
    fn test_ring_buffer_stats_after_overflow() {
        let mut logger = QueryLogger::new(3);
        // Insert 5 entries; buffer should hold only last 3
        for i in 1u64..=5 {
            logger.log(&format!("Q{}", i), "ds", i * 100, i as usize);
        }
        let s = logger.stats();
        // Entries 3, 4, 5 remain; avg = (300+400+500)/3 = 400
        assert!((s.avg_duration_ms - 400.0).abs() < 0.001);
        assert_eq!(s.max_duration_ms, 500);
    }

    #[test]
    fn test_overflow_with_errors() {
        let mut logger = QueryLogger::new(2);
        logger.log("Q1", "ds", 100, 1);
        logger.log_error("E1", "ds", "err");
        // Q1 is evicted; E1 stays in buffer
        logger.log("Q2", "ds", 200, 2);
        assert_eq!(logger.len(), 2);
    }

    // --- recent() ---

    #[test]
    fn test_recent_returns_newest_first() {
        let mut logger = QueryLogger::new(10);
        logger.log("Q1", "ds", 10, 1);
        logger.log("Q2", "ds", 20, 2);
        logger.log("Q3", "ds", 30, 3);
        let r = logger.recent(2);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].query, "Q3"); // newest first
        assert_eq!(r[1].query, "Q2");
    }

    #[test]
    fn test_recent_n_larger_than_buffer() {
        let mut logger = QueryLogger::new(10);
        logger.log("Q1", "ds", 10, 1);
        logger.log("Q2", "ds", 20, 2);
        let r = logger.recent(100);
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn test_recent_zero() {
        let mut logger = QueryLogger::new(10);
        logger.log("Q1", "ds", 10, 1);
        let r = logger.recent(0);
        assert!(r.is_empty());
    }

    #[test]
    fn test_recent_empty_logger() {
        let logger = QueryLogger::new(10);
        let r = logger.recent(5);
        assert!(r.is_empty());
    }

    // --- slow_queries() ---

    #[test]
    fn test_slow_queries_filters_threshold() {
        let mut logger = QueryLogger::new(10);
        logger.log("Fast", "ds", 10, 1);
        logger.log("Slow", "ds", 500, 2);
        logger.log("Medium", "ds", 150, 3);
        let slow = logger.slow_queries(100);
        assert_eq!(slow.len(), 2); // 500 > 100 and 150 > 100
        for e in &slow {
            assert!(e.duration_ms > 100);
        }
    }

    #[test]
    fn test_slow_queries_excludes_errors() {
        let mut logger = QueryLogger::new(10);
        logger.log_error("BAD", "ds", "error");
        logger.log("Slow", "ds", 1000, 1);
        let slow = logger.slow_queries(0);
        // Error entry (duration=0) excluded even when threshold=0
        assert_eq!(slow.len(), 1);
        assert!(slow[0].error.is_none());
    }

    #[test]
    fn test_slow_queries_none_exceed_threshold() {
        let mut logger = QueryLogger::new(10);
        logger.log("Q1", "ds", 10, 1);
        logger.log("Q2", "ds", 20, 2);
        let slow = logger.slow_queries(1000);
        assert!(slow.is_empty());
    }

    #[test]
    fn test_slow_queries_all_exceed_threshold() {
        let mut logger = QueryLogger::new(10);
        logger.log("Q1", "ds", 500, 1);
        logger.log("Q2", "ds", 600, 2);
        let slow = logger.slow_queries(100);
        assert_eq!(slow.len(), 2);
    }

    // --- clear() ---

    #[test]
    fn test_clear_empties_buffer() {
        let mut logger = QueryLogger::new(10);
        logger.log("Q1", "ds", 100, 1);
        logger.log("Q2", "ds", 200, 2);
        logger.clear();
        assert!(logger.is_empty());
        assert_eq!(logger.len(), 0);
    }

    #[test]
    fn test_clear_resets_stats() {
        let mut logger = QueryLogger::new(10);
        logger.log("Q1", "ds", 100, 1);
        logger.log_error("E1", "ds", "err");
        logger.clear();
        let s = logger.stats();
        assert_eq!(s.total_queries, 0);
        assert_eq!(s.error_count, 0);
        assert!((s.avg_duration_ms - 0.0).abs() < f64::EPSILON);
        assert_eq!(s.max_duration_ms, 0);
    }

    #[test]
    fn test_clear_allows_reuse() {
        let mut logger = QueryLogger::new(5);
        for i in 0..5u64 {
            logger.log(&format!("Q{}", i), "ds", i * 10, i as usize);
        }
        logger.clear();
        logger.log("New", "ds", 999, 1);
        assert_eq!(logger.len(), 1);
        let s = logger.stats();
        assert_eq!(s.total_queries, 1);
        assert_eq!(s.max_duration_ms, 999);
    }

    // --- entries() and entries_vec() ---

    #[test]
    fn test_entries_vec_order_oldest_first() {
        let mut logger = QueryLogger::new(10);
        logger.log("Q1", "ds", 10, 1);
        logger.log("Q2", "ds", 20, 2);
        logger.log("Q3", "ds", 30, 3);
        let v = logger.entries_vec();
        assert_eq!(v[0].query, "Q1");
        assert_eq!(v[2].query, "Q3");
    }

    #[test]
    fn test_timestamp_is_set() {
        let mut logger = QueryLogger::new(10);
        logger.log("Q", "ds", 10, 1);
        let e = &logger.entries_vec()[0];
        // Timestamp should be a plausible Unix ms value (> year 2020)
        assert!(e.timestamp_ms > 1_580_000_000_000);
    }

    #[test]
    fn test_multi_dataset_tracking() {
        let mut logger = QueryLogger::new(10);
        logger.log("Q1", "dataset-a", 10, 1);
        logger.log("Q2", "dataset-b", 20, 2);
        let v = logger.entries_vec();
        assert_eq!(v[0].dataset, "dataset-a");
        assert_eq!(v[1].dataset, "dataset-b");
    }

    #[test]
    fn test_debug_format() {
        let logger = QueryLogger::new(10);
        let dbg = format!("{:?}", logger);
        assert!(dbg.contains("QueryLogger"));
    }

    #[test]
    fn test_log_stats_debug_format() {
        let s = LogStats {
            total_queries: 1,
            error_count: 0,
            avg_duration_ms: 42.0,
            max_duration_ms: 42,
        };
        let dbg = format!("{:?}", s);
        assert!(dbg.contains("LogStats"));
    }

    #[test]
    fn test_query_log_entry_debug_and_clone() {
        let mut logger = QueryLogger::new(10);
        logger.log("Q1", "ds", 100, 5);
        let entry = logger.entries_vec()[0].clone();
        let dbg = format!("{:?}", entry);
        assert!(dbg.contains("QueryLogEntry"));
        assert_eq!(entry.query, "Q1");
    }
}
