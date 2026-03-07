//! Federated result streaming.
//!
//! Provides incremental delivery of SPARQL results from multiple federated
//! endpoints. Key features:
//!
//! - Incremental result delivery: results are yielded as they arrive from each
//!   source endpoint rather than waiting for all sources to complete.
//! - Streaming join: merge rows from multiple sources on-the-fly using a
//!   shared variable as the join key.
//! - Backpressure handling: a configurable buffer cap causes the streamer to
//!   mark upstream sources as paused when the buffer is full.
//! - Result ordering: sort rows by a named variable after collection.
//! - Partial result handling: deliver buffered rows before all sources complete.
//! - Stream statistics: rows delivered, total latency, buffer depth.
//! - Per-source timeout: don't wait indefinitely for a slow endpoint.
//! - Stream cancellation: cancel mid-stream and return partial results.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Row type
// ─────────────────────────────────────────────────────────────────────────────

/// A single SPARQL result row: variable name → string value.
pub type Row = HashMap<String, String>;

// ─────────────────────────────────────────────────────────────────────────────
// Source state
// ─────────────────────────────────────────────────────────────────────────────

/// Lifecycle state of a single federated source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceState {
    /// Source is actively producing rows.
    Active,
    /// Source is paused due to downstream backpressure.
    Paused,
    /// Source has been fully consumed.
    Completed,
    /// Source timed out before completing.
    TimedOut,
    /// Source failed with an error.
    Failed(String),
}

/// A federated source: an endpoint delivering rows incrementally.
#[derive(Debug)]
pub struct FederatedSource {
    /// Endpoint identifier (e.g. URL or name).
    pub endpoint_id: String,
    /// Pending rows not yet merged into the stream buffer.
    pub pending_rows: Vec<Row>,
    /// Current lifecycle state.
    pub state: SourceState,
    /// Simulated latency-so-far in milliseconds (set by the caller).
    pub latency_ms: u64,
    /// Timeout threshold in milliseconds (0 = no timeout).
    pub timeout_ms: u64,
}

impl FederatedSource {
    /// Create a new active source.
    pub fn new(endpoint_id: impl Into<String>, timeout_ms: u64) -> Self {
        Self {
            endpoint_id: endpoint_id.into(),
            pending_rows: Vec::new(),
            state: SourceState::Active,
            latency_ms: 0,
            timeout_ms,
        }
    }

    /// Push rows into the source as if they arrived from the endpoint.
    pub fn push_rows(&mut self, rows: impl IntoIterator<Item = Row>) {
        if self.state == SourceState::Active || self.state == SourceState::Paused {
            self.pending_rows.extend(rows);
        }
    }

    /// Mark the source as completed (no more rows will arrive).
    pub fn complete(&mut self) {
        if self.state == SourceState::Active || self.state == SourceState::Paused {
            self.state = SourceState::Completed;
        }
    }

    /// Check whether the source has exceeded its timeout.
    pub fn check_timeout(&mut self) {
        if self.timeout_ms > 0
            && self.latency_ms >= self.timeout_ms
            && self.state == SourceState::Active
        {
            self.state = SourceState::TimedOut;
        }
    }

    /// Whether this source is still producing or waiting.
    pub fn is_live(&self) -> bool {
        matches!(self.state, SourceState::Active | SourceState::Paused)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stream statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Running statistics for the result stream.
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    /// Total rows delivered to the consumer so far.
    pub rows_delivered: usize,
    /// Cumulative source latency sum (ms) across all sources.
    pub total_latency_ms: u64,
    /// Current number of rows in the internal buffer.
    pub buffer_depth: usize,
    /// Number of times the stream was paused due to backpressure.
    pub backpressure_events: usize,
    /// Number of sources that timed out.
    pub timed_out_sources: usize,
    /// Whether the stream was cancelled mid-flight.
    pub cancelled: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Join specification
// ─────────────────────────────────────────────────────────────────────────────

/// Specification for a streaming join across sources.
#[derive(Debug, Clone)]
pub struct JoinSpec {
    /// The variable name used as the join key (must appear in all rows).
    pub join_variable: String,
    /// Maximum time to wait for late-arriving matches (ms).
    pub join_timeout_ms: u64,
}

impl JoinSpec {
    pub fn new(join_variable: impl Into<String>, join_timeout_ms: u64) -> Self {
        Self {
            join_variable: join_variable.into(),
            join_timeout_ms,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result streamer
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`ResultStreamer`].
#[derive(Debug, Clone)]
pub struct StreamerConfig {
    /// Maximum number of buffered rows before backpressure is applied.
    pub buffer_cap: usize,
    /// Sort output rows by this variable name (empty = no sort).
    pub order_by: String,
    /// Whether to deliver partial results when a source times out or fails.
    pub partial_results: bool,
}

impl Default for StreamerConfig {
    fn default() -> Self {
        Self {
            buffer_cap: 1_000,
            order_by: String::new(),
            partial_results: true,
        }
    }
}

/// Manages incremental row delivery from multiple federated sources.
pub struct ResultStreamer {
    sources: Vec<FederatedSource>,
    buffer: Vec<Row>,
    config: StreamerConfig,
    stats: StreamStats,
    cancelled: bool,
}

impl ResultStreamer {
    /// Create a new streamer with the given configuration.
    pub fn new(config: StreamerConfig) -> Self {
        Self {
            sources: Vec::new(),
            buffer: Vec::new(),
            config,
            stats: StreamStats::default(),
            cancelled: false,
        }
    }

    /// Add a federated source to the streamer.
    pub fn add_source(&mut self, source: FederatedSource) {
        self.sources.push(source);
    }

    /// Number of registered sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Cancel the stream. Subsequent `drain` calls will return partial results.
    pub fn cancel(&mut self) {
        self.cancelled = true;
        self.stats.cancelled = true;
    }

    /// Whether the stream has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    /// Drain pending rows from all live sources into the internal buffer,
    /// applying backpressure when the buffer reaches capacity.
    ///
    /// Call this after pushing rows into sources to move them to the buffer.
    pub fn pull_from_sources(&mut self) {
        if self.cancelled {
            return;
        }

        for source in self.sources.iter_mut() {
            // Check timeout before pulling.
            source.check_timeout();

            if source.state == SourceState::TimedOut {
                self.stats.timed_out_sources += 1;
                // Rows already in pending are lost on timeout — by design.
                source.pending_rows.clear();
                source.state = SourceState::Completed;
                continue;
            }

            // Skip sources with no pending rows that are not live.
            // Completed sources may still have pending rows that arrived before
            // complete() was called — drain those before skipping.
            let has_pending = !source.pending_rows.is_empty();
            let is_active_or_paused =
                source.state == SourceState::Active || source.state == SourceState::Paused;
            let is_completed_with_pending = source.state == SourceState::Completed && has_pending;

            if !is_active_or_paused && !is_completed_with_pending {
                continue;
            }

            // Apply backpressure: pause active source if buffer is already full.
            if self.buffer.len() >= self.config.buffer_cap {
                if source.state == SourceState::Active {
                    source.state = SourceState::Paused;
                    self.stats.backpressure_events += 1;
                }
                continue;
            }

            // Resume a paused source when buffer has room.
            if source.state == SourceState::Paused {
                source.state = SourceState::Active;
            }

            // Track whether this is the first non-zero drain from this source
            // so we accumulate latency once per pull call per source.
            let had_rows = !source.pending_rows.is_empty();

            // Move pending rows to the buffer, up to buffer cap.
            let room = self.config.buffer_cap.saturating_sub(self.buffer.len());
            let take = room.min(source.pending_rows.len());
            let rows: Vec<Row> = source.pending_rows.drain(..take).collect();
            if had_rows {
                self.stats.total_latency_ms += source.latency_ms;
            }
            self.buffer.extend(rows);

            // After draining, if the buffer is now full and the source still has
            // pending rows, apply backpressure.
            if self.buffer.len() >= self.config.buffer_cap
                && !source.pending_rows.is_empty()
                && source.state == SourceState::Active
            {
                source.state = SourceState::Paused;
                self.stats.backpressure_events += 1;
            }
        }

        self.stats.buffer_depth = self.buffer.len();
    }

    /// Deliver all buffered rows to the consumer, clearing the buffer.
    ///
    /// If `order_by` is configured, rows are sorted by that variable's value
    /// (lexicographically) before delivery.
    ///
    /// Returns the delivered rows.
    pub fn drain(&mut self) -> Vec<Row> {
        if !self.config.order_by.is_empty() {
            let key = self.config.order_by.clone();
            self.buffer.sort_by(|a, b| {
                let va = a.get(&key).map(String::as_str).unwrap_or("");
                let vb = b.get(&key).map(String::as_str).unwrap_or("");
                va.cmp(vb)
            });
        }

        let rows = std::mem::take(&mut self.buffer);
        self.stats.rows_delivered += rows.len();
        self.stats.buffer_depth = 0;
        rows
    }

    /// Whether all sources have completed (or timed out / failed).
    pub fn all_sources_done(&self) -> bool {
        self.sources.iter().all(|s| !s.is_live())
    }

    /// Perform a streaming join of rows from all sources on `spec.join_variable`.
    ///
    /// Rows from different sources that share the same join-key value are
    /// merged into a single output row (later sources extend earlier columns).
    /// Unmatched rows are included as-is when `partial_results` is true.
    pub fn streaming_join(&mut self, spec: &JoinSpec) -> Vec<Row> {
        // Pull latest rows from sources.
        self.pull_from_sources();
        let rows = std::mem::take(&mut self.buffer);

        // Group rows by join key.
        let mut groups: HashMap<String, Row> = HashMap::new();
        let mut unmatched: Vec<Row> = Vec::new();

        for row in rows {
            match row.get(&spec.join_variable).cloned() {
                Some(key) => {
                    let entry = groups.entry(key).or_default();
                    // Merge all columns into the existing row.
                    for (col, val) in row {
                        entry.insert(col, val);
                    }
                }
                None => {
                    unmatched.push(row);
                }
            }
        }

        let mut result: Vec<Row> = groups.into_values().collect();
        if self.config.partial_results {
            result.extend(unmatched);
        }

        // Sort if ordered.
        if !self.config.order_by.is_empty() {
            let key = self.config.order_by.clone();
            result.sort_by(|a, b| {
                let va = a.get(&key).map(String::as_str).unwrap_or("");
                let vb = b.get(&key).map(String::as_str).unwrap_or("");
                va.cmp(vb)
            });
        }

        self.stats.rows_delivered += result.len();
        result
    }

    /// Return a snapshot of the current stream statistics.
    pub fn stats(&self) -> &StreamStats {
        &self.stats
    }

    /// Return the source at `index`, if it exists.
    pub fn source(&self, index: usize) -> Option<&FederatedSource> {
        self.sources.get(index)
    }

    /// Return a mutable reference to the source at `index`, if it exists.
    pub fn source_mut(&mut self, index: usize) -> Option<&mut FederatedSource> {
        self.sources.get_mut(index)
    }
}

impl Default for ResultStreamer {
    fn default() -> Self {
        Self::new(StreamerConfig::default())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(pairs: &[(&str, &str)]) -> Row {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn streamer_with_cap(cap: usize) -> ResultStreamer {
        ResultStreamer::new(StreamerConfig {
            buffer_cap: cap,
            order_by: String::new(),
            partial_results: true,
        })
    }

    // ── incremental delivery ─────────────────────────────────────────────────

    #[test]
    fn test_pull_and_drain_single_source() {
        let mut streamer = streamer_with_cap(100);
        let mut src = FederatedSource::new("ep1", 0);
        src.push_rows([make_row(&[("?s", "a"), ("?p", "b")])]);
        src.complete();
        streamer.add_source(src);
        streamer.pull_from_sources();
        let rows = streamer.drain();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["?s"], "a");
    }

    #[test]
    fn test_pull_from_two_sources() {
        let mut streamer = streamer_with_cap(100);
        let mut src1 = FederatedSource::new("ep1", 0);
        src1.push_rows([make_row(&[("?s", "s1")])]);
        src1.complete();
        let mut src2 = FederatedSource::new("ep2", 0);
        src2.push_rows([make_row(&[("?s", "s2")])]);
        src2.complete();
        streamer.add_source(src1);
        streamer.add_source(src2);
        streamer.pull_from_sources();
        let rows = streamer.drain();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_rows_delivered_stat_updated() {
        let mut streamer = streamer_with_cap(100);
        let mut src = FederatedSource::new("ep", 0);
        src.push_rows(vec![make_row(&[("?x", "1")]), make_row(&[("?x", "2")])]);
        src.complete();
        streamer.add_source(src);
        streamer.pull_from_sources();
        streamer.drain();
        assert_eq!(streamer.stats().rows_delivered, 2);
    }

    // ── backpressure ─────────────────────────────────────────────────────────

    #[test]
    fn test_backpressure_pauses_source_when_buffer_full() {
        let mut streamer = streamer_with_cap(2);
        let mut src = FederatedSource::new("ep", 0);
        src.push_rows([
            make_row(&[("?x", "1")]),
            make_row(&[("?x", "2")]),
            make_row(&[("?x", "3")]),
        ]);
        streamer.add_source(src);
        streamer.pull_from_sources();
        // Buffer should be capped at 2; source should be paused.
        assert_eq!(streamer.stats().buffer_depth, 2);
        assert_eq!(streamer.stats().backpressure_events, 1);
        assert_eq!(streamer.sources[0].state, SourceState::Paused);
    }

    #[test]
    fn test_backpressure_resumes_after_drain() {
        let mut streamer = streamer_with_cap(2);
        let mut src = FederatedSource::new("ep", 0);
        src.push_rows([
            make_row(&[("?x", "1")]),
            make_row(&[("?x", "2")]),
            make_row(&[("?x", "3")]),
        ]);
        streamer.add_source(src);
        streamer.pull_from_sources();
        // Drain to free the buffer.
        streamer.drain();
        // Now pull again — source should resume and deliver the third row.
        streamer.pull_from_sources();
        let rows = streamer.drain();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["?x"], "3");
    }

    // ── result ordering ──────────────────────────────────────────────────────

    #[test]
    fn test_order_by_variable() {
        let mut streamer = ResultStreamer::new(StreamerConfig {
            buffer_cap: 100,
            order_by: "?s".to_string(),
            partial_results: true,
        });
        let mut src = FederatedSource::new("ep", 0);
        src.push_rows([
            make_row(&[("?s", "charlie")]),
            make_row(&[("?s", "alice")]),
            make_row(&[("?s", "bob")]),
        ]);
        src.complete();
        streamer.add_source(src);
        streamer.pull_from_sources();
        let rows = streamer.drain();
        assert_eq!(rows[0]["?s"], "alice");
        assert_eq!(rows[1]["?s"], "bob");
        assert_eq!(rows[2]["?s"], "charlie");
    }

    // ── timeout ──────────────────────────────────────────────────────────────

    #[test]
    fn test_source_timeout_marks_completed() {
        let mut streamer = streamer_with_cap(100);
        let mut src = FederatedSource::new("slow-ep", 500);
        src.latency_ms = 600; // Exceeds timeout.
        src.push_rows([make_row(&[("?x", "1")])]);
        streamer.add_source(src);
        streamer.pull_from_sources();
        assert_eq!(streamer.stats().timed_out_sources, 1);
        assert_eq!(streamer.sources[0].state, SourceState::Completed);
    }

    #[test]
    fn test_source_below_timeout_not_timed_out() {
        let mut streamer = streamer_with_cap(100);
        let mut src = FederatedSource::new("ep", 1000);
        src.latency_ms = 500; // Below timeout.
        src.push_rows([make_row(&[("?x", "ok")])]);
        src.complete();
        streamer.add_source(src);
        streamer.pull_from_sources();
        assert_eq!(streamer.stats().timed_out_sources, 0);
    }

    #[test]
    fn test_zero_timeout_never_expires() {
        let mut streamer = streamer_with_cap(100);
        let mut src = FederatedSource::new("ep", 0); // 0 = no timeout
        src.latency_ms = 999_999;
        src.push_rows([make_row(&[("?x", "x")])]);
        src.complete();
        streamer.add_source(src);
        streamer.pull_from_sources();
        assert_eq!(streamer.stats().timed_out_sources, 0);
    }

    // ── cancellation ─────────────────────────────────────────────────────────

    #[test]
    fn test_cancel_stops_pull() {
        let mut streamer = streamer_with_cap(100);
        let mut src = FederatedSource::new("ep", 0);
        src.push_rows([make_row(&[("?x", "1")])]);
        streamer.add_source(src);
        streamer.cancel();
        streamer.pull_from_sources();
        assert!(streamer.is_cancelled());
        assert_eq!(streamer.stats().buffer_depth, 0);
    }

    #[test]
    fn test_cancel_flag_in_stats() {
        let mut streamer = streamer_with_cap(100);
        streamer.cancel();
        assert!(streamer.stats().cancelled);
    }

    // ── all_sources_done ─────────────────────────────────────────────────────

    #[test]
    fn test_all_sources_done_when_all_completed() {
        let mut streamer = streamer_with_cap(100);
        let mut src = FederatedSource::new("ep", 0);
        src.complete();
        streamer.add_source(src);
        assert!(streamer.all_sources_done());
    }

    #[test]
    fn test_not_all_done_if_one_active() {
        let mut streamer = streamer_with_cap(100);
        let src1 = FederatedSource::new("ep1", 0); // Active
        let mut src2 = FederatedSource::new("ep2", 0);
        src2.complete();
        streamer.add_source(src1);
        streamer.add_source(src2);
        assert!(!streamer.all_sources_done());
    }

    // ── streaming join ───────────────────────────────────────────────────────

    #[test]
    fn test_streaming_join_merges_rows_by_key() {
        let mut streamer = streamer_with_cap(100);
        let mut src1 = FederatedSource::new("ep1", 0);
        src1.push_rows([make_row(&[("?s", "alice"), ("?name", "Alice")])]);
        src1.complete();
        let mut src2 = FederatedSource::new("ep2", 0);
        src2.push_rows([make_row(&[("?s", "alice"), ("?age", "30")])]);
        src2.complete();
        streamer.add_source(src1);
        streamer.add_source(src2);

        let spec = JoinSpec::new("?s", 1000);
        let result = streamer.streaming_join(&spec);
        assert_eq!(result.len(), 1);
        let row = &result[0];
        assert_eq!(row["?s"], "alice");
        assert_eq!(row["?name"], "Alice");
        assert_eq!(row["?age"], "30");
    }

    #[test]
    fn test_streaming_join_different_keys_no_merge() {
        let mut streamer = streamer_with_cap(100);
        let mut src1 = FederatedSource::new("ep1", 0);
        src1.push_rows([make_row(&[("?s", "alice")])]);
        src1.complete();
        let mut src2 = FederatedSource::new("ep2", 0);
        src2.push_rows([make_row(&[("?s", "bob")])]);
        src2.complete();
        streamer.add_source(src1);
        streamer.add_source(src2);

        let spec = JoinSpec::new("?s", 1000);
        let result = streamer.streaming_join(&spec);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_streaming_join_unmatched_row_partial_results() {
        let mut streamer = ResultStreamer::new(StreamerConfig {
            buffer_cap: 100,
            order_by: String::new(),
            partial_results: true,
        });
        let mut src = FederatedSource::new("ep", 0);
        // Row without the join variable.
        src.push_rows([make_row(&[("?other", "x")])]);
        src.complete();
        streamer.add_source(src);

        let spec = JoinSpec::new("?s", 1000);
        let result = streamer.streaming_join(&spec);
        // Unmatched row included in partial mode.
        assert_eq!(result.len(), 1);
    }

    // ── partial results ──────────────────────────────────────────────────────

    #[test]
    fn test_partial_results_from_incomplete_sources() {
        let mut streamer = streamer_with_cap(100);
        let mut src1 = FederatedSource::new("ep1", 0);
        src1.push_rows([make_row(&[("?x", "a")])]);
        src1.complete();
        // src2 is still active (never completed).
        let mut src2 = FederatedSource::new("ep2", 0);
        src2.push_rows([make_row(&[("?x", "b")])]);
        streamer.add_source(src1);
        streamer.add_source(src2);
        streamer.pull_from_sources();
        let rows = streamer.drain();
        // Both rows delivered even though src2 is not complete.
        assert_eq!(rows.len(), 2);
    }

    // ── buffer depth stat ────────────────────────────────────────────────────

    #[test]
    fn test_buffer_depth_stat_reset_after_drain() {
        let mut streamer = streamer_with_cap(100);
        let mut src = FederatedSource::new("ep", 0);
        src.push_rows([make_row(&[("?x", "1")]), make_row(&[("?x", "2")])]);
        src.complete();
        streamer.add_source(src);
        streamer.pull_from_sources();
        assert_eq!(streamer.stats().buffer_depth, 2);
        streamer.drain();
        assert_eq!(streamer.stats().buffer_depth, 0);
    }

    // ── source state helpers ─────────────────────────────────────────────────

    #[test]
    fn test_source_is_live_when_active() {
        let src = FederatedSource::new("ep", 0);
        assert!(src.is_live());
    }

    #[test]
    fn test_source_not_live_when_completed() {
        let mut src = FederatedSource::new("ep", 0);
        src.complete();
        assert!(!src.is_live());
    }

    #[test]
    fn test_source_push_rows_ignored_when_completed() {
        let mut src = FederatedSource::new("ep", 0);
        src.complete();
        src.push_rows([make_row(&[("?x", "1")])]);
        assert!(src.pending_rows.is_empty());
    }

    #[test]
    fn test_source_count() {
        let mut streamer = streamer_with_cap(100);
        streamer.add_source(FederatedSource::new("ep1", 0));
        streamer.add_source(FederatedSource::new("ep2", 0));
        assert_eq!(streamer.source_count(), 2);
    }

    #[test]
    fn test_source_accessor() {
        let mut streamer = streamer_with_cap(100);
        streamer.add_source(FederatedSource::new("ep-alpha", 0));
        assert_eq!(
            streamer.source(0).map(|s| s.endpoint_id.as_str()),
            Some("ep-alpha")
        );
        assert!(streamer.source(99).is_none());
    }

    #[test]
    fn test_default_streamer_has_no_sources() {
        let streamer = ResultStreamer::default();
        assert_eq!(streamer.source_count(), 0);
        assert!(streamer.all_sources_done());
    }

    #[test]
    fn test_drain_empty_buffer_returns_empty() {
        let mut streamer = streamer_with_cap(100);
        let rows = streamer.drain();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_stats_total_latency_accumulated() {
        let mut streamer = streamer_with_cap(100);
        let mut src = FederatedSource::new("ep", 0);
        src.latency_ms = 42;
        src.push_rows([make_row(&[("?x", "v")])]);
        src.complete();
        streamer.add_source(src);
        streamer.pull_from_sources();
        assert_eq!(streamer.stats().total_latency_ms, 42);
    }

    #[test]
    fn test_join_spec_fields() {
        let spec = JoinSpec::new("?s", 5000);
        assert_eq!(spec.join_variable, "?s");
        assert_eq!(spec.join_timeout_ms, 5000);
    }

    #[test]
    fn test_failed_source_not_live() {
        let mut src = FederatedSource::new("ep", 0);
        src.state = SourceState::Failed("oops".into());
        assert!(!src.is_live());
    }

    #[test]
    fn test_source_mut_allows_modification() {
        let mut streamer = streamer_with_cap(100);
        streamer.add_source(FederatedSource::new("ep", 0));
        if let Some(src) = streamer.source_mut(0) {
            src.latency_ms = 77;
        }
        assert_eq!(streamer.source(0).map(|s| s.latency_ms), Some(77));
    }

    #[test]
    fn test_multiple_drain_calls_accumulate_rows_delivered() {
        let mut streamer = streamer_with_cap(100);
        let mut src = FederatedSource::new("ep", 0);
        src.push_rows([make_row(&[("?x", "1")])]);
        streamer.add_source(src);
        streamer.pull_from_sources();
        streamer.drain();
        // Push more rows.
        if let Some(src) = streamer.source_mut(0) {
            src.state = SourceState::Active;
            src.push_rows([make_row(&[("?x", "2")]), make_row(&[("?x", "3")])]);
        }
        streamer.pull_from_sources();
        streamer.drain();
        assert_eq!(streamer.stats().rows_delivered, 3);
    }

    // ── additional edge cases ────────────────────────────────────────────────

    #[test]
    fn test_streamer_with_no_sources_all_done() {
        let streamer = streamer_with_cap(10);
        assert!(streamer.all_sources_done());
    }

    #[test]
    fn test_source_complete_marks_not_live() {
        let mut src = FederatedSource::new("ep", 0);
        assert!(src.is_live());
        src.complete();
        assert!(!src.is_live());
    }

    #[test]
    fn test_backpressure_event_count_on_second_pull() {
        let mut streamer = streamer_with_cap(1);
        let mut src = FederatedSource::new("ep", 0);
        src.push_rows([make_row(&[("?x", "a")]), make_row(&[("?x", "b")])]);
        streamer.add_source(src);
        streamer.pull_from_sources();
        // One backpressure event because second row couldn't fit.
        assert_eq!(streamer.stats().backpressure_events, 1);
    }

    #[test]
    fn test_order_by_missing_key_sorts_empty_string() {
        // Rows missing the order-by key should be treated as empty string.
        let mut streamer = ResultStreamer::new(StreamerConfig {
            buffer_cap: 100,
            order_by: "?missing".to_string(),
            partial_results: true,
        });
        let mut src = FederatedSource::new("ep", 0);
        src.push_rows([make_row(&[("?x", "b")]), make_row(&[("?x", "a")])]);
        src.complete();
        streamer.add_source(src);
        streamer.pull_from_sources();
        let rows = streamer.drain();
        // All rows have empty string for ?missing, so order is stable (no crash).
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_source_failed_state_not_live() {
        let mut src = FederatedSource::new("ep", 0);
        src.state = SourceState::Failed("timeout".into());
        assert!(!src.is_live());
    }

    #[test]
    fn test_join_multiple_keys_all_merge() {
        let mut streamer = streamer_with_cap(100);
        let mut src1 = FederatedSource::new("ep1", 0);
        src1.push_rows([
            make_row(&[("?s", "x"), ("?a", "alpha")]),
            make_row(&[("?s", "y"), ("?a", "beta")]),
        ]);
        src1.complete();
        let mut src2 = FederatedSource::new("ep2", 0);
        src2.push_rows([
            make_row(&[("?s", "x"), ("?b", "1")]),
            make_row(&[("?s", "y"), ("?b", "2")]),
        ]);
        src2.complete();
        streamer.add_source(src1);
        streamer.add_source(src2);
        let spec = JoinSpec::new("?s", 1000);
        let result = streamer.streaming_join(&spec);
        assert_eq!(result.len(), 2);
        let x_row = result
            .iter()
            .find(|r| r.get("?s").map(|v| v == "x").unwrap_or(false));
        assert!(x_row.is_some());
        let x_row = x_row.expect("x row must exist");
        assert_eq!(x_row.get("?a").map(String::as_str), Some("alpha"));
        assert_eq!(x_row.get("?b").map(String::as_str), Some("1"));
    }

    #[test]
    fn test_pull_does_nothing_when_cancelled() {
        let mut streamer = streamer_with_cap(100);
        let mut src = FederatedSource::new("ep", 0);
        src.push_rows([make_row(&[("?x", "1")])]);
        streamer.add_source(src);
        streamer.cancel();
        streamer.pull_from_sources();
        assert_eq!(streamer.stats().buffer_depth, 0);
        assert_eq!(streamer.stats().rows_delivered, 0);
    }

    #[test]
    fn test_timeout_clears_pending_rows() {
        let mut streamer = streamer_with_cap(100);
        let mut src = FederatedSource::new("ep", 100);
        src.latency_ms = 200; // Exceeds timeout of 100.
        src.push_rows([make_row(&[("?x", "lost")])]);
        streamer.add_source(src);
        streamer.pull_from_sources();
        // Pending rows should have been cleared on timeout.
        let rows = streamer.drain();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_source_state_equality() {
        assert_eq!(SourceState::Active, SourceState::Active);
        assert_ne!(SourceState::Active, SourceState::Completed);
        assert_eq!(
            SourceState::Failed("x".into()),
            SourceState::Failed("x".into())
        );
    }

    #[test]
    fn test_streamer_config_defaults() {
        let cfg = StreamerConfig::default();
        assert_eq!(cfg.buffer_cap, 1_000);
        assert!(cfg.order_by.is_empty());
        assert!(cfg.partial_results);
    }

    #[test]
    fn test_stream_stats_default() {
        let stats = StreamStats::default();
        assert_eq!(stats.rows_delivered, 0);
        assert_eq!(stats.backpressure_events, 0);
        assert!(!stats.cancelled);
    }

    #[test]
    fn test_streaming_join_ordered_by_key() {
        let mut streamer = ResultStreamer::new(StreamerConfig {
            buffer_cap: 100,
            order_by: "?s".to_string(),
            partial_results: true,
        });
        let mut src = FederatedSource::new("ep", 0);
        src.push_rows([make_row(&[("?s", "charlie")]), make_row(&[("?s", "alice")])]);
        src.complete();
        streamer.add_source(src);
        let spec = JoinSpec::new("?s", 0);
        let result = streamer.streaming_join(&spec);
        assert_eq!(result[0]["?s"], "alice");
        assert_eq!(result[1]["?s"], "charlie");
    }

    #[test]
    fn test_timed_out_source_contributes_zero_rows() {
        let mut streamer = streamer_with_cap(100);
        let mut src = FederatedSource::new("ep", 50);
        src.latency_ms = 100;
        src.push_rows([make_row(&[("?x", "never")])]);
        streamer.add_source(src);
        streamer.pull_from_sources();
        let rows = streamer.drain();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_source_endpoint_id_preserved() {
        let src = FederatedSource::new("my-endpoint", 0);
        assert_eq!(src.endpoint_id, "my-endpoint");
    }

    #[test]
    fn test_paused_state_is_live() {
        let mut src = FederatedSource::new("ep", 0);
        src.state = SourceState::Paused;
        assert!(src.is_live());
    }
}
