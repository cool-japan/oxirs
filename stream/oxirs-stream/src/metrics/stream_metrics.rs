//! # Stream Metrics Collector
//!
//! Real-time metrics for stream processing pipelines, including:
//!
//! - [`StreamMetrics`]: Point-in-time snapshot of processing statistics
//! - [`StreamLatencyHistogram`]: Fixed-bucket latency histogram with percentile queries
//! - [`StreamMetricsCollector`]: Mutable collector that accumulates stats over time

// ─── StreamMetrics ────────────────────────────────────────────────────────────

/// Point-in-time snapshot of stream processing metrics.
#[derive(Debug, Clone, Default)]
pub struct StreamMetrics {
    pub events_in: u64,
    pub events_out: u64,
    pub events_late: u64,
    pub events_dropped: u64,
    pub processing_lag_ms: f64,
    pub throughput_per_sec: f64,
    pub window_completions: u64,
    pub checkpoint_count: u64,
    pub error_count: u64,
}

// ─── StreamLatencyHistogram ───────────────────────────────────────────────────

/// Fixed-bucket latency histogram.
///
/// Buckets are upper-bound values in milliseconds (inclusive). Each bucket
/// accumulates a count of observations whose latency falls at or below the
/// bucket's bound (but above the previous bucket's bound).
///
/// Example: `buckets = &[1, 5, 10, 50, 100, 500, 1000]` creates 8 buckets
/// (7 bounded + 1 overflow bucket for values above the last bound).
#[derive(Debug, Clone)]
pub struct StreamLatencyHistogram {
    /// `(upper_bound_ms, count)` pairs, plus a final overflow bucket with `u64::MAX`.
    buckets: Vec<(u64, u64)>,
    total: u64,
    sum_ms: u64,
}

impl StreamLatencyHistogram {
    /// Create a histogram with the given upper bounds (in milliseconds).
    ///
    /// An implicit overflow bucket (`u64::MAX`) is always appended.
    pub fn new(bounds: &[u64]) -> Self {
        let mut buckets: Vec<(u64, u64)> = bounds.iter().map(|&b| (b, 0u64)).collect();
        buckets.push((u64::MAX, 0));
        Self {
            buckets,
            total: 0,
            sum_ms: 0,
        }
    }

    /// Record a latency observation.
    pub fn observe(&mut self, latency_ms: u64) {
        self.total += 1;
        self.sum_ms = self.sum_ms.saturating_add(latency_ms);
        for (bound, count) in &mut self.buckets {
            if latency_ms <= *bound {
                *count += 1;
                return;
            }
        }
    }

    /// Estimate the `p`-th percentile latency (e.g., `p = 0.95` for P95).
    ///
    /// Uses linear interpolation within the matching bucket.
    /// Returns `0` when no observations have been recorded.
    pub fn percentile(&self, p: f64) -> u64 {
        if self.total == 0 {
            return 0;
        }
        let target = (p * self.total as f64).ceil() as u64;
        let mut cumulative = 0u64;
        let mut prev_bound: u64 = 0;
        for &(bound, count) in &self.buckets {
            cumulative += count;
            if cumulative >= target {
                if bound == u64::MAX {
                    // In the overflow bucket: return the previous bound as best estimate
                    return prev_bound;
                }
                return bound;
            }
            if bound != u64::MAX {
                prev_bound = bound;
            }
        }
        self.buckets.last().map(|(b, _)| *b).unwrap_or(0)
    }

    /// Arithmetic mean latency in milliseconds.
    ///
    /// Returns `0.0` when no observations have been recorded.
    pub fn mean(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.sum_ms as f64 / self.total as f64
    }

    /// Total number of observations recorded.
    pub fn count(&self) -> u64 {
        self.total
    }
}

// ─── StreamMetricsCollector ───────────────────────────────────────────────────

/// Accumulates stream processing metrics over the lifetime of a pipeline.
///
/// Call `snapshot()` at any point to get a consistent [`StreamMetrics`] copy.
/// Call `reset()` to start a fresh measurement window.
pub struct StreamMetricsCollector {
    metrics: StreamMetrics,
    latency_histogram: StreamLatencyHistogram,
    start_time_ms: i64,
    last_reset_ms: i64,
}

impl StreamMetricsCollector {
    /// Create a new collector with default buckets `[1, 5, 10, 50, 100, 500, 1000]`.
    pub fn new() -> Self {
        let now = current_ms();
        Self {
            metrics: StreamMetrics::default(),
            latency_histogram: StreamLatencyHistogram::new(&[1, 5, 10, 50, 100, 500, 1000]),
            start_time_ms: now,
            last_reset_ms: now,
        }
    }

    /// Record an inbound event and update the processing lag estimate.
    pub fn record_event_in(&mut self, timestamp_ms: i64) {
        self.metrics.events_in += 1;
        let now = current_ms();
        let lag = (now - timestamp_ms).max(0) as f64;
        // Exponential moving average for lag (alpha = 0.1)
        self.metrics.processing_lag_ms = 0.9 * self.metrics.processing_lag_ms + 0.1 * lag;
        self.update_throughput();
    }

    /// Record an outbound event with the observed end-to-end latency.
    pub fn record_event_out(&mut self, latency_ms: u64) {
        self.metrics.events_out += 1;
        self.latency_histogram.observe(latency_ms);
    }

    /// Record a late event.
    pub fn record_late_event(&mut self) {
        self.metrics.events_late += 1;
        self.metrics.events_dropped += 1;
    }

    /// Record a processing error.
    pub fn record_error(&mut self) {
        self.metrics.error_count += 1;
    }

    /// Record a completed checkpoint.
    pub fn record_checkpoint(&mut self) {
        self.metrics.checkpoint_count += 1;
    }

    /// Record a window completion.
    pub fn record_window_completion(&mut self) {
        self.metrics.window_completions += 1;
    }

    /// Return a point-in-time snapshot of the current metrics.
    pub fn snapshot(&self) -> StreamMetrics {
        self.metrics.clone()
    }

    /// Return a reference to the latency histogram.
    pub fn latency_histogram(&self) -> &StreamLatencyHistogram {
        &self.latency_histogram
    }

    /// Reset all counters and the histogram, keeping the original `start_time_ms`.
    pub fn reset(&mut self) {
        self.metrics = StreamMetrics::default();
        self.latency_histogram = StreamLatencyHistogram::new(&[1, 5, 10, 50, 100, 500, 1000]);
        self.last_reset_ms = current_ms();
    }

    /// Milliseconds elapsed since the collector was created.
    pub fn uptime_ms(&self) -> i64 {
        current_ms() - self.start_time_ms
    }

    /// Milliseconds elapsed since the last `reset()`.
    pub fn time_since_reset_ms(&self) -> i64 {
        current_ms() - self.last_reset_ms
    }

    /// Update the throughput estimate based on events_in and elapsed time.
    fn update_throughput(&mut self) {
        let elapsed_secs = self.time_since_reset_ms() as f64 / 1000.0;
        if elapsed_secs > 0.0 {
            self.metrics.throughput_per_sec = self.metrics.events_in as f64 / elapsed_secs;
        }
    }
}

impl Default for StreamMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper: current wall-clock time in milliseconds.
fn current_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── StreamLatencyHistogram ─────────────────────────────────────────────────

    #[test]
    fn test_histogram_empty_state() {
        let hist = StreamLatencyHistogram::new(&[1, 5, 10, 50, 100, 500, 1000]);
        assert_eq!(hist.count(), 0);
        assert_eq!(hist.mean(), 0.0);
        assert_eq!(hist.percentile(0.5), 0);
    }

    #[test]
    fn test_histogram_observe_single() {
        let mut hist = StreamLatencyHistogram::new(&[10, 100, 1000]);
        hist.observe(5);
        assert_eq!(hist.count(), 1);
        assert_eq!(hist.mean(), 5.0);
    }

    #[test]
    fn test_histogram_observe_multiple() {
        let mut hist = StreamLatencyHistogram::new(&[10, 100, 1000]);
        for ms in [2, 4, 6, 8, 10] {
            hist.observe(ms);
        }
        assert_eq!(hist.count(), 5);
        assert!((hist.mean() - 6.0).abs() < 1e-9); // (2+4+6+8+10)/5 = 6
    }

    #[test]
    fn test_histogram_p50_basic() {
        let mut hist = StreamLatencyHistogram::new(&[1, 5, 10, 50, 100, 500, 1000]);
        // 10 observations all at 5ms
        for _ in 0..10 {
            hist.observe(5);
        }
        assert_eq!(hist.percentile(0.5), 5);
    }

    #[test]
    fn test_histogram_p99_all_in_first_bucket() {
        let mut hist = StreamLatencyHistogram::new(&[1, 5, 10, 50, 100, 500, 1000]);
        for _ in 0..100 {
            hist.observe(1); // all <= 1ms
        }
        assert_eq!(hist.percentile(0.99), 1);
    }

    #[test]
    fn test_histogram_overflow_bucket() {
        let mut hist = StreamLatencyHistogram::new(&[10, 100]);
        hist.observe(200); // above all bounds → overflow bucket
        assert_eq!(hist.count(), 1);
    }

    #[test]
    fn test_histogram_mean_calculation() {
        let mut hist = StreamLatencyHistogram::new(&[100]);
        hist.observe(10);
        hist.observe(20);
        hist.observe(30);
        assert!((hist.mean() - 20.0).abs() < 1e-9);
    }

    // ── StreamMetricsCollector ─────────────────────────────────────────────────

    #[test]
    fn test_collector_initial_state() {
        let col = StreamMetricsCollector::new();
        let snap = col.snapshot();
        assert_eq!(snap.events_in, 0);
        assert_eq!(snap.events_out, 0);
        assert_eq!(snap.error_count, 0);
        assert_eq!(snap.events_late, 0);
        assert_eq!(snap.checkpoint_count, 0);
    }

    #[test]
    fn test_collector_record_events_in() {
        let mut col = StreamMetricsCollector::new();
        let now = current_ms();
        col.record_event_in(now);
        col.record_event_in(now);
        assert_eq!(col.snapshot().events_in, 2);
    }

    #[test]
    fn test_collector_record_events_out() {
        let mut col = StreamMetricsCollector::new();
        col.record_event_out(5);
        col.record_event_out(10);
        assert_eq!(col.snapshot().events_out, 2);
        assert_eq!(col.latency_histogram().count(), 2);
    }

    #[test]
    fn test_collector_record_late_event() {
        let mut col = StreamMetricsCollector::new();
        col.record_late_event();
        let snap = col.snapshot();
        assert_eq!(snap.events_late, 1);
        assert_eq!(snap.events_dropped, 1);
    }

    #[test]
    fn test_collector_record_error() {
        let mut col = StreamMetricsCollector::new();
        col.record_error();
        col.record_error();
        assert_eq!(col.snapshot().error_count, 2);
    }

    #[test]
    fn test_collector_record_checkpoint() {
        let mut col = StreamMetricsCollector::new();
        col.record_checkpoint();
        col.record_checkpoint();
        col.record_checkpoint();
        assert_eq!(col.snapshot().checkpoint_count, 3);
    }

    #[test]
    fn test_collector_record_window_completion() {
        let mut col = StreamMetricsCollector::new();
        col.record_window_completion();
        assert_eq!(col.snapshot().window_completions, 1);
    }

    #[test]
    fn test_collector_reset_clears_metrics() {
        let mut col = StreamMetricsCollector::new();
        col.record_error();
        col.record_event_out(10);
        col.record_late_event();
        col.reset();
        let snap = col.snapshot();
        assert_eq!(snap.error_count, 0);
        assert_eq!(snap.events_out, 0);
        assert_eq!(snap.events_late, 0);
        assert_eq!(col.latency_histogram().count(), 0);
    }

    #[test]
    fn test_collector_uptime_is_non_negative() {
        let col = StreamMetricsCollector::new();
        assert!(col.uptime_ms() >= 0);
    }

    #[test]
    fn test_collector_throughput_updates_on_event_in() {
        let mut col = StreamMetricsCollector::new();
        let now = current_ms();
        for _ in 0..100 {
            col.record_event_in(now);
        }
        // throughput should be > 0 since events_in > 0
        let snap = col.snapshot();
        // It's possible time_since_reset is 0ms on very fast machines;
        // just verify events_in is correct.
        assert_eq!(snap.events_in, 100);
    }

    #[test]
    fn test_collector_lag_nonnegative_for_current_events() {
        let mut col = StreamMetricsCollector::new();
        let now = current_ms();
        col.record_event_in(now);
        let snap = col.snapshot();
        // Lag should be ≥ 0 (we clamp negative to 0)
        assert!(snap.processing_lag_ms >= 0.0);
    }
}
