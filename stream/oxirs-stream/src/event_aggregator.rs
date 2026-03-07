//! Event stream aggregation: tumbling, sliding, and session windows.
//!
//! Provides `EventAggregator` for in-memory buffering of timestamped events
//! and on-demand aggregation into `AggWindow` summaries grouped by key.

use std::collections::{HashMap, VecDeque};

/// A single timestamped numeric event.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    pub key: String,
    pub value: f64,
    pub timestamp_ms: i64,
}

impl Event {
    /// Create a new event.
    pub fn new(key: impl Into<String>, value: f64, timestamp_ms: i64) -> Self {
        Self {
            key: key.into(),
            value,
            timestamp_ms,
        }
    }
}

/// An aggregated window result.
#[derive(Debug, Clone, PartialEq)]
pub struct AggWindow {
    pub key: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub count: usize,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
    pub avg: f64,
}

impl AggWindow {
    fn new(key: String, start_ms: i64, end_ms: i64) -> Self {
        Self {
            key,
            start_ms,
            end_ms,
            count: 0,
            sum: 0.0,
            min: f64::MAX,
            max: f64::MIN,
            avg: 0.0,
        }
    }

    fn add(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
        self.avg = self.sum / self.count as f64;
    }

    fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Window type for aggregation.
#[derive(Debug, Clone, PartialEq)]
pub enum WindowType {
    /// Fixed-size non-overlapping windows of `window_ms` milliseconds.
    Tumbling(i64),
    /// Overlapping windows: each window has `size_ms` and advances by `step_ms`.
    Sliding { size_ms: i64, step_ms: i64 },
    /// Dynamic windows: a window closes when no event arrives within `gap_ms`.
    Session(i64),
}

/// Aggregates buffered events into time windows.
pub struct EventAggregator {
    window: WindowType,
    buffer: VecDeque<Event>,
}

impl EventAggregator {
    /// Create a new aggregator with the given window type.
    pub fn new(window: WindowType) -> Self {
        Self {
            window,
            buffer: VecDeque::new(),
        }
    }

    /// Push an event into the buffer (need not be in order, but ordering improves accuracy).
    pub fn push(&mut self, event: Event) {
        self.buffer.push_back(event);
    }

    /// Return finalized (closed) windows up to `now_ms`.
    ///
    /// For tumbling windows every completed bucket is returned.
    /// For sliding windows every completed step window is returned.
    /// For session windows every closed session (gap detected) is returned.
    pub fn aggregate(&self, now_ms: i64) -> Vec<AggWindow> {
        let events: Vec<&Event> = self.buffer.iter().collect();
        match &self.window {
            WindowType::Tumbling(window_ms) => tumbling_aggregate_ref(&events, *window_ms, now_ms),
            WindowType::Sliding { size_ms, step_ms } => {
                sliding_aggregate_ref(&events, *size_ms, *step_ms, now_ms)
            }
            WindowType::Session(gap_ms) => session_aggregate_ref(&events, *gap_ms),
        }
    }

    /// Aggregate and group by key, returning one `AggWindow` per key (using the latest window).
    pub fn aggregate_by_key(&self, now_ms: i64) -> HashMap<String, AggWindow> {
        let windows = self.aggregate(now_ms);
        let mut map: HashMap<String, AggWindow> = HashMap::new();
        for w in windows {
            let entry = map.entry(w.key.clone()).or_insert_with(|| w.clone());
            // keep the one with the latest end_ms
            if w.end_ms > entry.end_ms {
                *entry = w;
            }
        }
        map
    }

    /// Remove all events with `timestamp_ms < cutoff_ms`. Returns count removed.
    pub fn flush_before(&mut self, cutoff_ms: i64) -> usize {
        let before = self.buffer.len();
        self.buffer.retain(|e| e.timestamp_ms >= cutoff_ms);
        before - self.buffer.len()
    }

    /// Number of buffered events.
    pub fn pending_count(&self) -> usize {
        self.buffer.len()
    }

    /// Earliest event timestamp in buffer, or `None` if empty.
    pub fn earliest_ms(&self) -> Option<i64> {
        self.buffer.iter().map(|e| e.timestamp_ms).min()
    }

    /// Latest event timestamp in buffer, or `None` if empty.
    pub fn latest_ms(&self) -> Option<i64> {
        self.buffer.iter().map(|e| e.timestamp_ms).max()
    }
}

// ---- standalone helpers (pub for tests) ------------------------------------

/// Compute tumbling window aggregates for a slice of events.
pub fn tumbling_aggregate(events: &[Event], window_ms: i64) -> Vec<AggWindow> {
    let refs: Vec<&Event> = events.iter().collect();
    tumbling_aggregate_ref(&refs, window_ms, i64::MAX)
}

/// Compute sliding window aggregates for a slice of events.
pub fn sliding_aggregate(events: &[Event], size_ms: i64, step_ms: i64) -> Vec<AggWindow> {
    let refs: Vec<&Event> = events.iter().collect();
    sliding_aggregate_ref(&refs, size_ms, step_ms, i64::MAX)
}

// ---- internal helpers -------------------------------------------------------

fn tumbling_aggregate_ref(events: &[&Event], window_ms: i64, now_ms: i64) -> Vec<AggWindow> {
    if events.is_empty() || window_ms <= 0 {
        return vec![];
    }

    let min_ts = events.iter().map(|e| e.timestamp_ms).min().unwrap_or(0);
    let max_ts = events.iter().map(|e| e.timestamp_ms).max().unwrap_or(0);

    // Closed windows only: window [start, start+window_ms) must be < now_ms
    let mut results = Vec::new();
    let mut bucket_start = floor_div(min_ts, window_ms) * window_ms;

    while bucket_start < now_ms {
        let bucket_end = bucket_start + window_ms;
        if bucket_end > now_ms {
            break; // still open
        }
        if bucket_start > max_ts {
            break;
        }

        let mut acc: HashMap<String, AggWindow> = HashMap::new();
        for e in events.iter() {
            if e.timestamp_ms >= bucket_start && e.timestamp_ms < bucket_end {
                acc.entry(e.key.clone())
                    .or_insert_with(|| AggWindow::new(e.key.clone(), bucket_start, bucket_end))
                    .add(e.value);
            }
        }
        for (_, w) in acc {
            if !w.is_empty() {
                results.push(w);
            }
        }
        bucket_start = bucket_end;
    }
    results
}

fn sliding_aggregate_ref(
    events: &[&Event],
    size_ms: i64,
    step_ms: i64,
    now_ms: i64,
) -> Vec<AggWindow> {
    if events.is_empty() || size_ms <= 0 || step_ms <= 0 {
        return vec![];
    }

    let min_ts = events.iter().map(|e| e.timestamp_ms).min().unwrap_or(0);
    let max_ts = events.iter().map(|e| e.timestamp_ms).max().unwrap_or(0);

    let mut results = Vec::new();
    let first_window_start = floor_div(min_ts, step_ms) * step_ms;
    let mut window_start = first_window_start;

    while window_start <= max_ts {
        let window_end = window_start + size_ms;
        if window_end > now_ms {
            break; // still open
        }

        let mut acc: HashMap<String, AggWindow> = HashMap::new();
        for e in events.iter() {
            if e.timestamp_ms >= window_start && e.timestamp_ms < window_end {
                acc.entry(e.key.clone())
                    .or_insert_with(|| AggWindow::new(e.key.clone(), window_start, window_end))
                    .add(e.value);
            }
        }
        for (_, w) in acc {
            if !w.is_empty() {
                results.push(w);
            }
        }
        window_start += step_ms;
    }
    results
}

fn session_aggregate_ref(events: &[&Event], gap_ms: i64) -> Vec<AggWindow> {
    if events.is_empty() || gap_ms <= 0 {
        return vec![];
    }

    // Group by key first, sort each group by timestamp
    let mut by_key: HashMap<String, Vec<i64>> = HashMap::new();
    let mut values_by_key: HashMap<String, Vec<f64>> = HashMap::new();
    for e in events.iter() {
        by_key.entry(e.key.clone()).or_default().push(e.timestamp_ms);
        values_by_key.entry(e.key.clone()).or_default().push(e.value);
    }

    // For each key build events sorted
    let mut key_events: HashMap<String, Vec<(i64, f64)>> = HashMap::new();
    for e in events.iter() {
        key_events
            .entry(e.key.clone())
            .or_default()
            .push((e.timestamp_ms, e.value));
    }

    let mut results = Vec::new();
    for (key, mut evts) in key_events {
        evts.sort_by_key(|(ts, _)| *ts);

        let mut session_start = evts[0].0;
        let mut last_ts = evts[0].0;
        let mut acc = AggWindow::new(key.clone(), session_start, last_ts);
        acc.add(evts[0].1);

        for &(ts, val) in evts[1..].iter() {
            if ts - last_ts >= gap_ms {
                // close the current session
                acc.end_ms = last_ts;
                results.push(acc);
                session_start = ts;
                acc = AggWindow::new(key.clone(), session_start, ts);
            }
            acc.add(val);
            last_ts = ts;
        }
        acc.end_ms = last_ts;
        results.push(acc);
    }
    results
}

/// Integer floor division (rounds towards negative infinity).
fn floor_div(a: i64, b: i64) -> i64 {
    let d = a / b;
    let r = a % b;
    if (r != 0) && ((r < 0) != (b < 0)) {
        d - 1
    } else {
        d
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(key: &str, value: f64, ts: i64) -> Event {
        Event::new(key, value, ts)
    }

    // --- Event creation -------------------------------------------------

    #[test]
    fn test_event_new() {
        let e = Event::new("sensor1", 42.0, 1000);
        assert_eq!(e.key, "sensor1");
        assert_eq!(e.value, 42.0);
        assert_eq!(e.timestamp_ms, 1000);
    }

    #[test]
    fn test_event_clone() {
        let e = ev("k", 1.0, 10);
        let c = e.clone();
        assert_eq!(e, c);
    }

    // --- AggWindow -------------------------------------------------------

    #[test]
    fn test_agg_window_initial_empty() {
        let w = AggWindow::new("k".into(), 0, 1000);
        assert!(w.is_empty());
    }

    #[test]
    fn test_agg_window_add_single() {
        let mut w = AggWindow::new("k".into(), 0, 1000);
        w.add(5.0);
        assert_eq!(w.count, 1);
        assert_eq!(w.sum, 5.0);
        assert_eq!(w.min, 5.0);
        assert_eq!(w.max, 5.0);
        assert_eq!(w.avg, 5.0);
    }

    #[test]
    fn test_agg_window_add_multiple() {
        let mut w = AggWindow::new("k".into(), 0, 1000);
        w.add(1.0);
        w.add(3.0);
        w.add(5.0);
        assert_eq!(w.count, 3);
        assert!((w.sum - 9.0).abs() < 1e-9);
        assert_eq!(w.min, 1.0);
        assert_eq!(w.max, 5.0);
        assert!((w.avg - 3.0).abs() < 1e-9);
    }

    // --- EventAggregator basic -------------------------------------------

    #[test]
    fn test_aggregator_new_empty() {
        let a = EventAggregator::new(WindowType::Tumbling(1000));
        assert_eq!(a.pending_count(), 0);
        assert_eq!(a.earliest_ms(), None);
        assert_eq!(a.latest_ms(), None);
    }

    #[test]
    fn test_push_increases_count() {
        let mut a = EventAggregator::new(WindowType::Tumbling(1000));
        a.push(ev("x", 1.0, 100));
        assert_eq!(a.pending_count(), 1);
        a.push(ev("y", 2.0, 200));
        assert_eq!(a.pending_count(), 2);
    }

    #[test]
    fn test_earliest_latest() {
        let mut a = EventAggregator::new(WindowType::Tumbling(1000));
        a.push(ev("k", 1.0, 500));
        a.push(ev("k", 2.0, 100));
        a.push(ev("k", 3.0, 800));
        assert_eq!(a.earliest_ms(), Some(100));
        assert_eq!(a.latest_ms(), Some(800));
    }

    #[test]
    fn test_flush_before() {
        let mut a = EventAggregator::new(WindowType::Tumbling(1000));
        a.push(ev("k", 1.0, 100));
        a.push(ev("k", 2.0, 500));
        a.push(ev("k", 3.0, 900));
        let removed = a.flush_before(500);
        assert_eq!(removed, 1);
        assert_eq!(a.pending_count(), 2);
    }

    #[test]
    fn test_flush_before_all() {
        let mut a = EventAggregator::new(WindowType::Tumbling(1000));
        a.push(ev("k", 1.0, 100));
        a.push(ev("k", 2.0, 200));
        let removed = a.flush_before(300);
        assert_eq!(removed, 2);
        assert_eq!(a.pending_count(), 0);
    }

    #[test]
    fn test_flush_before_none() {
        let mut a = EventAggregator::new(WindowType::Tumbling(1000));
        a.push(ev("k", 1.0, 100));
        let removed = a.flush_before(50);
        assert_eq!(removed, 0);
        assert_eq!(a.pending_count(), 1);
    }

    // --- Tumbling windows ------------------------------------------------

    #[test]
    fn test_tumbling_no_closed_window() {
        let mut a = EventAggregator::new(WindowType::Tumbling(1000));
        a.push(ev("k", 1.0, 500));
        // now_ms = 1000 means window [0,1000) is not closed yet (bucket_end == now_ms)
        let w = a.aggregate(1000);
        assert!(w.is_empty(), "window not closed yet");
    }

    #[test]
    fn test_tumbling_one_closed_window() {
        let mut a = EventAggregator::new(WindowType::Tumbling(1000));
        a.push(ev("k", 5.0, 100));
        a.push(ev("k", 3.0, 700));
        // now_ms = 1001 closes window [0, 1000)
        let windows = a.aggregate(1001);
        let w = windows.iter().find(|w| w.key == "k").expect("window found");
        assert_eq!(w.start_ms, 0);
        assert_eq!(w.end_ms, 1000);
        assert_eq!(w.count, 2);
        assert!((w.sum - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_tumbling_two_windows() {
        let events = vec![
            ev("k", 1.0, 100),
            ev("k", 2.0, 500),
            ev("k", 3.0, 1100),
            ev("k", 4.0, 1800),
        ];
        let windows = tumbling_aggregate(&events, 1000);
        assert_eq!(windows.len(), 2);
    }

    #[test]
    fn test_tumbling_multi_key() {
        let events = vec![
            ev("a", 1.0, 100),
            ev("b", 2.0, 200),
            ev("a", 3.0, 300),
        ];
        let windows = tumbling_aggregate(&events, 1000);
        // window [0,1000) is not closed when now=MAX, so all are closed
        let a_win = windows.iter().find(|w| w.key == "a").expect("key a");
        assert_eq!(a_win.count, 2);
        let b_win = windows.iter().find(|w| w.key == "b").expect("key b");
        assert_eq!(b_win.count, 1);
    }

    #[test]
    fn test_tumbling_empty_events() {
        let windows = tumbling_aggregate(&[], 1000);
        assert!(windows.is_empty());
    }

    #[test]
    fn test_tumbling_zero_window_ms() {
        let events = vec![ev("k", 1.0, 100)];
        let windows = tumbling_aggregate(&events, 0);
        assert!(windows.is_empty());
    }

    #[test]
    fn test_tumbling_min_max_correct() {
        let events = vec![
            ev("k", 10.0, 100),
            ev("k", 1.0, 200),
            ev("k", 5.0, 300),
        ];
        let windows = tumbling_aggregate(&events, 1000);
        let w = &windows[0];
        assert_eq!(w.min, 1.0);
        assert_eq!(w.max, 10.0);
    }

    // --- Sliding windows -------------------------------------------------

    #[test]
    fn test_sliding_basic() {
        let events = vec![
            ev("k", 1.0, 0),
            ev("k", 2.0, 500),
            ev("k", 3.0, 1000),
            ev("k", 4.0, 1500),
        ];
        // size=1000, step=500 → windows [0,1000), [500,1500), [1000,2000)...
        let windows = sliding_aggregate(&events, 1000, 500);
        assert!(!windows.is_empty());
    }

    #[test]
    fn test_sliding_empty() {
        let windows = sliding_aggregate(&[], 1000, 500);
        assert!(windows.is_empty());
    }

    #[test]
    fn test_sliding_zero_size() {
        let events = vec![ev("k", 1.0, 100)];
        let windows = sliding_aggregate(&events, 0, 500);
        assert!(windows.is_empty());
    }

    #[test]
    fn test_sliding_zero_step() {
        let events = vec![ev("k", 1.0, 100)];
        let windows = sliding_aggregate(&events, 1000, 0);
        assert!(windows.is_empty());
    }

    #[test]
    fn test_sliding_overlap() {
        // Event at t=750 should appear in both [0,1000) and [500,1500) windows
        let events = vec![ev("k", 7.0, 750)];
        let windows = sliding_aggregate(&events, 1000, 500);
        let count: usize = windows.iter().map(|w| w.count).sum();
        // 750 is in [0,1000) and [500,1500)
        assert_eq!(count, 2);
    }

    // --- Session windows -------------------------------------------------

    #[test]
    fn test_session_single_event() {
        let events = vec![ev("k", 5.0, 1000)];
        let a = EventAggregator::new(WindowType::Session(500));
        let mut agg = EventAggregator::new(WindowType::Session(500));
        for e in events {
            agg.push(e);
        }
        let _ = a; // silence
        let windows = agg.aggregate(9999);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].count, 1);
        assert_eq!(windows[0].sum, 5.0);
    }

    #[test]
    fn test_session_two_sessions() {
        let mut a = EventAggregator::new(WindowType::Session(100));
        a.push(ev("k", 1.0, 0));
        a.push(ev("k", 2.0, 50));
        // gap of 500 > 100 ms
        a.push(ev("k", 3.0, 550));
        a.push(ev("k", 4.0, 600));
        let windows = a.aggregate(9999);
        assert_eq!(windows.len(), 2);
    }

    #[test]
    fn test_session_no_gap_single_session() {
        let mut a = EventAggregator::new(WindowType::Session(1000));
        a.push(ev("k", 1.0, 0));
        a.push(ev("k", 2.0, 200));
        a.push(ev("k", 3.0, 400));
        let windows = a.aggregate(9999);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].count, 3);
    }

    #[test]
    fn test_session_empty() {
        let a = EventAggregator::new(WindowType::Session(500));
        let windows = a.aggregate(9999);
        assert!(windows.is_empty());
    }

    // --- aggregate_by_key -----------------------------------------------

    #[test]
    fn test_aggregate_by_key_two_keys() {
        let mut a = EventAggregator::new(WindowType::Tumbling(1000));
        a.push(ev("sensor_a", 10.0, 100));
        a.push(ev("sensor_b", 20.0, 200));
        a.push(ev("sensor_a", 30.0, 300));
        let map = a.aggregate_by_key(2000);
        assert!(map.contains_key("sensor_a"));
        assert!(map.contains_key("sensor_b"));
        assert_eq!(map["sensor_a"].count, 2);
        assert_eq!(map["sensor_b"].count, 1);
    }

    #[test]
    fn test_aggregate_by_key_empty() {
        let a = EventAggregator::new(WindowType::Tumbling(1000));
        let map = a.aggregate_by_key(9999);
        assert!(map.is_empty());
    }

    // --- floor_div -------------------------------------------------------

    #[test]
    fn test_floor_div_positive() {
        assert_eq!(floor_div(1500, 1000), 1);
        assert_eq!(floor_div(1000, 1000), 1);
        assert_eq!(floor_div(999, 1000), 0);
    }

    #[test]
    fn test_floor_div_negative() {
        assert_eq!(floor_div(-1, 1000), -1);
        assert_eq!(floor_div(-1000, 1000), -1);
        assert_eq!(floor_div(-1001, 1000), -2);
    }

    // --- Integration / edge cases ----------------------------------------

    #[test]
    fn test_large_dataset_tumbling() {
        let mut a = EventAggregator::new(WindowType::Tumbling(1000));
        for i in 0..100 {
            a.push(ev("sensor", i as f64, i * 100));
        }
        // All events span 0..9900 ms. Windows up to now_ms=100000 should all close.
        let windows = a.aggregate(100_000);
        assert!(!windows.is_empty());
    }

    #[test]
    fn test_pending_count_after_flush() {
        let mut a = EventAggregator::new(WindowType::Tumbling(1000));
        for i in 0..10 {
            a.push(ev("k", 1.0, i * 100));
        }
        assert_eq!(a.pending_count(), 10);
        a.flush_before(500);
        assert!(a.pending_count() < 10);
    }

    #[test]
    fn test_window_type_clone() {
        let wt = WindowType::Sliding {
            size_ms: 1000,
            step_ms: 500,
        };
        let c = wt.clone();
        assert_eq!(wt, c);
    }

    #[test]
    fn test_session_gap_ms_zero_empty() {
        let events = vec![ev("k", 1.0, 100), ev("k", 2.0, 200)];
        let mut a = EventAggregator::new(WindowType::Session(0));
        for e in events {
            a.push(e);
        }
        // gap_ms = 0, session_aggregate_ref returns empty
        let w = a.aggregate(9999);
        assert!(w.is_empty());
    }

    #[test]
    fn test_tumbling_aggregate_sum_correct() {
        let events: Vec<Event> = (0..5).map(|i| ev("k", 2.0, i * 100)).collect();
        let windows = tumbling_aggregate(&events, 1000);
        assert_eq!(windows.len(), 1);
        assert!((windows[0].sum - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_sliding_multiple_keys() {
        let events = vec![
            ev("a", 1.0, 100),
            ev("b", 2.0, 200),
            ev("a", 3.0, 300),
            ev("b", 4.0, 400),
        ];
        let windows = sliding_aggregate(&events, 1000, 500);
        assert!(windows.iter().any(|w| w.key == "a"));
        assert!(windows.iter().any(|w| w.key == "b"));
    }
}
