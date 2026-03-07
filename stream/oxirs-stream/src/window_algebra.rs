//! # Stream Windowing Algebra
//!
//! Provides a composable windowing algebra for stream processing with support for
//! tumbling, sliding, session, and count-based windows. Integrates with the
//! existing watermark infrastructure for late-data handling and eviction.
//!
//! ## Features
//!
//! - **Tumbling windows**: Fixed-size, non-overlapping time intervals
//! - **Sliding windows**: Fixed-size intervals that advance by a configurable slide
//! - **Session windows**: Gap-based windows that merge when events arrive within a timeout
//! - **Count-based windows**: Windows that trigger after N events
//! - **Watermark-based eviction**: Automatically close and evict windows past the watermark
//! - **Late data handling**: Configurable policies for events arriving after window close
//! - **Window aggregation**: Composable fold/reduce over window contents

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::Duration;

// ─────────────────────────────────────────────
// Window types
// ─────────────────────────────────────────────

/// The kind of window to apply.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WindowKind {
    /// Fixed-size, non-overlapping time window.
    Tumbling {
        /// Window size.
        size: Duration,
    },
    /// Fixed-size window that advances by `slide`.
    Sliding {
        /// Window size.
        size: Duration,
        /// Slide interval.
        slide: Duration,
    },
    /// Gap-based window: a new window opens when no events arrive within `gap`.
    Session {
        /// Inactivity gap that closes the current session.
        gap: Duration,
    },
    /// Trigger after receiving `count` events.
    Count {
        /// Number of events per window.
        count: usize,
    },
}

/// Policy for handling events that arrive after the window has been closed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LatePolicy {
    /// Silently drop late events.
    #[default]
    Drop,
    /// Accept into a side output.
    SideOutput,
    /// Reopen the window (allowed lateness).
    AllowedLateness {
        /// How long past the window end to still accept events.
        lateness: Duration,
    },
}

/// Configuration for the window algebra operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowAlgebraConfig {
    /// The kind of window.
    pub kind: WindowKind,
    /// Policy for late-arriving events.
    pub late_policy: LatePolicy,
    /// Maximum number of open windows before forced eviction of the oldest.
    pub max_open_windows: usize,
    /// Whether to emit partial results on eviction.
    pub emit_on_evict: bool,
}

impl Default for WindowAlgebraConfig {
    fn default() -> Self {
        Self {
            kind: WindowKind::Tumbling {
                size: Duration::from_secs(60),
            },
            late_policy: LatePolicy::default(),
            max_open_windows: 10_000,
            emit_on_evict: true,
        }
    }
}

// ─────────────────────────────────────────────
// Window identifier and pane
// ─────────────────────────────────────────────

/// Identifies a window instance.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct WindowId {
    /// Start of window (millis since epoch) or sequence number.
    pub start: i64,
    /// End of window (millis since epoch) or sequence number.
    pub end: i64,
    /// Optional key for keyed windows.
    pub key: Option<String>,
}

impl WindowId {
    /// Create a time-range window id.
    pub fn time_range(start_ms: i64, end_ms: i64) -> Self {
        Self {
            start: start_ms,
            end: end_ms,
            key: None,
        }
    }

    /// Create a keyed time-range window id.
    pub fn keyed(start_ms: i64, end_ms: i64, key: impl Into<String>) -> Self {
        Self {
            start: start_ms,
            end: end_ms,
            key: Some(key.into()),
        }
    }

    /// Duration of this window in milliseconds.
    pub fn duration_ms(&self) -> i64 {
        self.end - self.start
    }

    /// Whether `ts` falls within this window.
    pub fn contains(&self, ts: i64) -> bool {
        ts >= self.start && ts < self.end
    }
}

/// A windowed event with its event-time timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowEvent<T: Clone> {
    /// The event payload.
    pub value: T,
    /// Event-time timestamp in milliseconds since epoch.
    pub timestamp_ms: i64,
    /// Ingestion time.
    pub ingestion_time: DateTime<Utc>,
}

impl<T: Clone> WindowEvent<T> {
    pub fn new(value: T, timestamp_ms: i64) -> Self {
        Self {
            value,
            timestamp_ms,
            ingestion_time: Utc::now(),
        }
    }
}

/// State of a single window pane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WindowState {
    /// Actively collecting events.
    Open,
    /// Closed but within allowed-lateness period.
    Closing,
    /// Fully closed and results emitted.
    Closed,
}

/// A single window pane containing events.
#[derive(Debug, Clone)]
pub struct WindowPane<T: Clone> {
    /// Window identifier.
    pub id: WindowId,
    /// Events in this pane.
    pub events: Vec<WindowEvent<T>>,
    /// Current state.
    pub state: WindowState,
    /// When the pane was created.
    pub created_at: DateTime<Utc>,
    /// Allowed lateness deadline (if applicable).
    pub lateness_deadline_ms: Option<i64>,
}

impl<T: Clone> WindowPane<T> {
    fn new(id: WindowId) -> Self {
        Self {
            id,
            events: Vec::new(),
            state: WindowState::Open,
            created_at: Utc::now(),
            lateness_deadline_ms: None,
        }
    }

    fn new_with_lateness(id: WindowId, lateness_ms: i64) -> Self {
        let deadline = id.end + lateness_ms;
        Self {
            id,
            events: Vec::new(),
            state: WindowState::Open,
            created_at: Utc::now(),
            lateness_deadline_ms: Some(deadline),
        }
    }

    /// Number of events in this pane.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether this pane is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Minimum event timestamp in this pane.
    pub fn min_timestamp(&self) -> Option<i64> {
        self.events.iter().map(|e| e.timestamp_ms).min()
    }

    /// Maximum event timestamp in this pane.
    pub fn max_timestamp(&self) -> Option<i64> {
        self.events.iter().map(|e| e.timestamp_ms).max()
    }
}

/// Emitted when a window fires (triggers).
#[derive(Debug, Clone)]
pub struct WindowOutput<T: Clone> {
    /// The window that fired.
    pub window_id: WindowId,
    /// Events in the window.
    pub events: Vec<WindowEvent<T>>,
    /// Whether this is a partial result (from eviction).
    pub is_partial: bool,
    /// Number of late events that were dropped.
    pub late_dropped: usize,
    /// Number of late events accepted into side output.
    pub late_side_output: usize,
}

/// Statistics for the window algebra operator.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WindowAlgebraStats {
    /// Total events processed.
    pub total_events: u64,
    /// Total windows opened.
    pub windows_opened: u64,
    /// Total windows closed.
    pub windows_closed: u64,
    /// Total windows evicted.
    pub windows_evicted: u64,
    /// Total late events dropped.
    pub late_events_dropped: u64,
    /// Total late events sent to side output.
    pub late_events_side_output: u64,
    /// Total late events accepted within allowed lateness.
    pub late_events_accepted: u64,
    /// Currently open windows.
    pub open_windows: u64,
}

// ─────────────────────────────────────────────
// WindowAlgebra operator
// ─────────────────────────────────────────────

/// The main stream windowing algebra operator.
///
/// Generic over the event payload type `T`.
pub struct WindowAlgebra<T: Clone> {
    config: WindowAlgebraConfig,
    /// Open window panes keyed by WindowId.
    panes: BTreeMap<WindowId, WindowPane<T>>,
    /// Side output for late events.
    side_output: VecDeque<WindowEvent<T>>,
    /// Current watermark (millis).
    watermark_ms: i64,
    /// Statistics.
    stats: WindowAlgebraStats,
    // For count windows: buffer keyed by key
    count_buffers: HashMap<Option<String>, Vec<WindowEvent<T>>>,
    // Monotonic sequence for count window ids
    count_seq: i64,
    // For session windows: last event time per key
    session_last_event: HashMap<Option<String>, i64>,
}

impl<T: Clone> WindowAlgebra<T> {
    /// Create a new windowing operator with the given configuration.
    pub fn new(config: WindowAlgebraConfig) -> Self {
        Self {
            config,
            panes: BTreeMap::new(),
            side_output: VecDeque::new(),
            watermark_ms: i64::MIN,
            stats: WindowAlgebraStats::default(),
            count_buffers: HashMap::new(),
            count_seq: 0,
            session_last_event: HashMap::new(),
        }
    }

    /// Create a tumbling window operator.
    pub fn tumbling(size: Duration) -> Self {
        Self::new(WindowAlgebraConfig {
            kind: WindowKind::Tumbling { size },
            ..Default::default()
        })
    }

    /// Create a sliding window operator.
    pub fn sliding(size: Duration, slide: Duration) -> Self {
        Self::new(WindowAlgebraConfig {
            kind: WindowKind::Sliding { size, slide },
            ..Default::default()
        })
    }

    /// Create a session window operator.
    pub fn session(gap: Duration) -> Self {
        Self::new(WindowAlgebraConfig {
            kind: WindowKind::Session { gap },
            ..Default::default()
        })
    }

    /// Create a count-based window operator.
    pub fn count(count: usize) -> Self {
        Self::new(WindowAlgebraConfig {
            kind: WindowKind::Count { count },
            ..Default::default()
        })
    }

    /// Set the late data policy.
    pub fn with_late_policy(mut self, policy: LatePolicy) -> Self {
        self.config.late_policy = policy;
        self
    }

    /// Set the maximum number of open windows.
    pub fn with_max_open_windows(mut self, max: usize) -> Self {
        self.config.max_open_windows = max;
        self
    }

    /// Get current statistics.
    pub fn stats(&self) -> &WindowAlgebraStats {
        &self.stats
    }

    /// Get current watermark in milliseconds.
    pub fn watermark_ms(&self) -> i64 {
        self.watermark_ms
    }

    /// Get the number of currently open panes.
    pub fn open_pane_count(&self) -> usize {
        self.panes
            .values()
            .filter(|p| p.state == WindowState::Open || p.state == WindowState::Closing)
            .count()
    }

    /// Get the side output (late events).
    pub fn drain_side_output(&mut self) -> Vec<WindowEvent<T>> {
        self.side_output.drain(..).collect()
    }

    /// Advance the watermark and return any windows that should be closed.
    pub fn advance_watermark(&mut self, watermark_ms: i64) -> Vec<WindowOutput<T>> {
        if watermark_ms <= self.watermark_ms {
            return Vec::new();
        }
        self.watermark_ms = watermark_ms;
        self.close_expired_windows()
    }

    /// Ingest a single event. Returns any triggered window outputs.
    pub fn ingest(&mut self, event: WindowEvent<T>) -> Vec<WindowOutput<T>> {
        self.stats.total_events += 1;

        match &self.config.kind {
            WindowKind::Tumbling { size } => self.ingest_tumbling(event, *size),
            WindowKind::Sliding { size, slide } => self.ingest_sliding(event, *size, *slide),
            WindowKind::Session { gap } => self.ingest_session(event, *gap),
            WindowKind::Count { count } => self.ingest_count(event, *count),
        }
    }

    /// Ingest a batch of events.
    pub fn ingest_batch(&mut self, events: Vec<WindowEvent<T>>) -> Vec<WindowOutput<T>> {
        let mut outputs = Vec::new();
        for event in events {
            outputs.extend(self.ingest(event));
        }
        outputs
    }

    // ─── Tumbling ────────────────────────────────────────

    fn ingest_tumbling(&mut self, event: WindowEvent<T>, size: Duration) -> Vec<WindowOutput<T>> {
        let size_ms = size.as_millis() as i64;
        if size_ms == 0 {
            return Vec::new();
        }
        let window_start = (event.timestamp_ms / size_ms) * size_ms;
        let window_end = window_start + size_ms;
        let wid = WindowId::time_range(window_start, window_end);

        self.assign_to_window(event, wid)
    }

    // ─── Sliding ─────────────────────────────────────────

    fn ingest_sliding(
        &mut self,
        event: WindowEvent<T>,
        size: Duration,
        slide: Duration,
    ) -> Vec<WindowOutput<T>> {
        let size_ms = size.as_millis() as i64;
        let slide_ms = slide.as_millis() as i64;
        if slide_ms == 0 || size_ms == 0 {
            return Vec::new();
        }

        // An event belongs to all windows whose [start, start+size) contains it
        let ts = event.timestamp_ms;
        // The earliest window start that could contain ts
        let latest_start = (ts / slide_ms) * slide_ms;
        let earliest_start = latest_start - size_ms + slide_ms;

        let mut outputs = Vec::new();
        let mut start = earliest_start;
        while start <= latest_start {
            let end = start + size_ms;
            if ts >= start && ts < end {
                let wid = WindowId::time_range(start, end);
                outputs.extend(self.assign_to_window(event.clone(), wid));
            }
            start += slide_ms;
        }
        outputs
    }

    // ─── Session ─────────────────────────────────────────

    fn ingest_session(&mut self, event: WindowEvent<T>, gap: Duration) -> Vec<WindowOutput<T>> {
        let gap_ms = gap.as_millis() as i64;
        let ts = event.timestamp_ms;
        let key: Option<String> = None; // non-keyed session

        let mut outputs = Vec::new();

        if let Some(&last_ts) = self.session_last_event.get(&key) {
            if ts - last_ts > gap_ms {
                // Gap exceeded: close the current session window
                outputs.extend(self.close_session_windows(&key));
            }
        }

        // Find or create the active session window for this key.
        // If an existing session is extended, migrate its pane to the new key.
        let active_wid = self.find_or_extend_active_session(&key, ts, gap_ms);
        self.session_last_event.insert(key.clone(), ts);

        outputs.extend(self.assign_to_window(event, active_wid));
        outputs
    }

    fn find_or_extend_active_session(
        &mut self,
        key: &Option<String>,
        ts: i64,
        gap_ms: i64,
    ) -> WindowId {
        // Look for an open session window whose key matches
        let existing_wid = self
            .panes
            .iter()
            .find(|(wid, pane)| {
                wid.key == *key && pane.state == WindowState::Open && ts >= wid.start
            })
            .map(|(wid, _)| wid.clone());

        if let Some(old_wid) = existing_wid {
            let new_end = ts + gap_ms;
            if new_end == old_wid.end {
                // No change needed, reuse existing window id
                return old_wid;
            }
            // Build the extended window id
            let new_wid = WindowId {
                start: old_wid.start,
                end: new_end,
                key: key.clone(),
            };
            // Migrate the pane from old key to new key so events accumulate
            // in a single pane rather than creating duplicates.
            if let Some(mut pane) = self.panes.remove(&old_wid) {
                pane.id = new_wid.clone();
                self.panes.insert(new_wid.clone(), pane);
            }
            new_wid
        } else {
            // No active session, create new
            WindowId {
                start: ts,
                end: ts + gap_ms,
                key: key.clone(),
            }
        }
    }

    fn close_session_windows(&mut self, key: &Option<String>) -> Vec<WindowOutput<T>> {
        let mut outputs = Vec::new();
        let wids_to_close: Vec<WindowId> = self
            .panes
            .keys()
            .filter(|wid| wid.key == *key)
            .cloned()
            .collect();

        for wid in wids_to_close {
            if let Some(mut pane) = self.panes.remove(&wid) {
                pane.state = WindowState::Closed;
                self.stats.windows_closed += 1;
                self.stats.open_windows = self.stats.open_windows.saturating_sub(1);
                outputs.push(WindowOutput {
                    window_id: wid,
                    events: pane.events,
                    is_partial: false,
                    late_dropped: 0,
                    late_side_output: 0,
                });
            }
        }
        outputs
    }

    // ─── Count-based ─────────────────────────────────────

    fn ingest_count(&mut self, event: WindowEvent<T>, count: usize) -> Vec<WindowOutput<T>> {
        let key: Option<String> = None;
        let buf = self.count_buffers.entry(key.clone()).or_default();
        buf.push(event);

        let mut outputs = Vec::new();
        while buf.len() >= count {
            let window_events: Vec<_> = buf.drain(..count).collect();
            let seq = self.count_seq;
            self.count_seq += 1;
            let wid = WindowId {
                start: seq * count as i64,
                end: (seq + 1) * count as i64,
                key: key.clone(),
            };
            self.stats.windows_opened += 1;
            self.stats.windows_closed += 1;
            outputs.push(WindowOutput {
                window_id: wid,
                events: window_events,
                is_partial: false,
                late_dropped: 0,
                late_side_output: 0,
            });
        }
        outputs
    }

    // ─── Common assignment ───────────────────────────────

    fn assign_to_window(&mut self, event: WindowEvent<T>, wid: WindowId) -> Vec<WindowOutput<T>> {
        let mut outputs = Vec::new();

        // Check if window is already closed
        if let Some(pane) = self.panes.get(&wid) {
            match pane.state {
                WindowState::Closed => {
                    return self.handle_late_event(event);
                }
                WindowState::Closing => {
                    // Check allowed lateness
                    if let Some(deadline) = pane.lateness_deadline_ms {
                        if event.timestamp_ms > deadline {
                            return self.handle_late_event(event);
                        }
                    }
                    // Fall through: accept event
                }
                WindowState::Open => {
                    // Normal case
                }
            }
        }

        // Check if event is before watermark and window doesn't exist yet
        if event.timestamp_ms < self.watermark_ms && !self.panes.contains_key(&wid) {
            return self.handle_late_event(event);
        }

        // Get or create pane
        if !self.panes.contains_key(&wid) {
            let pane = match self.config.late_policy {
                LatePolicy::AllowedLateness { lateness } => {
                    WindowPane::new_with_lateness(wid.clone(), lateness.as_millis() as i64)
                }
                _ => WindowPane::new(wid.clone()),
            };
            self.panes.insert(wid.clone(), pane);
            self.stats.windows_opened += 1;
            self.stats.open_windows += 1;
        }

        if let Some(pane) = self.panes.get_mut(&wid) {
            pane.events.push(event);
        }

        // Enforce max open windows
        outputs.extend(self.enforce_max_open_windows());

        outputs
    }

    fn handle_late_event(&mut self, event: WindowEvent<T>) -> Vec<WindowOutput<T>> {
        match self.config.late_policy {
            LatePolicy::Drop => {
                self.stats.late_events_dropped += 1;
            }
            LatePolicy::SideOutput => {
                self.stats.late_events_side_output += 1;
                self.side_output.push_back(event);
            }
            LatePolicy::AllowedLateness { .. } => {
                // If we're here, it's past the allowed lateness too
                self.stats.late_events_dropped += 1;
            }
        }
        Vec::new()
    }

    fn close_expired_windows(&mut self) -> Vec<WindowOutput<T>> {
        let mut outputs = Vec::new();
        let wm = self.watermark_ms;

        let expired: Vec<WindowId> = self
            .panes
            .iter()
            .filter(|(wid, pane)| {
                if pane.state == WindowState::Closed {
                    return false;
                }
                match pane.lateness_deadline_ms {
                    Some(deadline) => wm >= deadline,
                    None => wm >= wid.end,
                }
            })
            .map(|(wid, _)| wid.clone())
            .collect();

        for wid in expired {
            if let Some(mut pane) = self.panes.remove(&wid) {
                pane.state = WindowState::Closed;
                self.stats.windows_closed += 1;
                self.stats.open_windows = self.stats.open_windows.saturating_sub(1);
                outputs.push(WindowOutput {
                    window_id: wid,
                    events: pane.events,
                    is_partial: false,
                    late_dropped: 0,
                    late_side_output: 0,
                });
            }
        }

        outputs
    }

    fn enforce_max_open_windows(&mut self) -> Vec<WindowOutput<T>> {
        let mut outputs = Vec::new();
        while self.panes.len() > self.config.max_open_windows {
            // Evict oldest window (smallest start)
            if let Some(wid) = self.panes.keys().next().cloned() {
                if let Some(mut pane) = self.panes.remove(&wid) {
                    pane.state = WindowState::Closed;
                    self.stats.windows_evicted += 1;
                    self.stats.open_windows = self.stats.open_windows.saturating_sub(1);
                    if self.config.emit_on_evict {
                        outputs.push(WindowOutput {
                            window_id: wid,
                            events: pane.events,
                            is_partial: true,
                            late_dropped: 0,
                            late_side_output: 0,
                        });
                    }
                }
            } else {
                break;
            }
        }
        outputs
    }

    /// Flush all open windows, emitting their contents.
    pub fn flush(&mut self) -> Vec<WindowOutput<T>> {
        let mut outputs = Vec::new();

        // Also flush count buffers
        for (key, buf) in self.count_buffers.drain() {
            if !buf.is_empty() {
                let seq = self.count_seq;
                self.count_seq += 1;
                let wid = WindowId {
                    start: seq * 1000,
                    end: (seq + 1) * 1000,
                    key,
                };
                outputs.push(WindowOutput {
                    window_id: wid,
                    events: buf,
                    is_partial: true,
                    late_dropped: 0,
                    late_side_output: 0,
                });
            }
        }

        let wids: Vec<_> = self.panes.keys().cloned().collect();
        for wid in wids {
            if let Some(mut pane) = self.panes.remove(&wid) {
                pane.state = WindowState::Closed;
                self.stats.windows_closed += 1;
                self.stats.open_windows = self.stats.open_windows.saturating_sub(1);
                outputs.push(WindowOutput {
                    window_id: wid,
                    events: pane.events,
                    is_partial: true,
                    late_dropped: 0,
                    late_side_output: 0,
                });
            }
        }
        outputs
    }

    /// Aggregate window contents using a fold function.
    pub fn aggregate<A, F>(&self, window_id: &WindowId, init: A, fold: F) -> Option<A>
    where
        F: Fn(A, &T) -> A,
    {
        self.panes.get(window_id).map(|pane| {
            pane.events
                .iter()
                .fold(init, |acc, evt| fold(acc, &evt.value))
        })
    }
}

// ─────────────────────────────────────────────
// Window Assigner (helper to compute window assignments)
// ─────────────────────────────────────────────

/// Computes which tumbling windows a given timestamp belongs to.
pub fn tumbling_window_for(ts_ms: i64, size: Duration) -> WindowId {
    let size_ms = size.as_millis() as i64;
    if size_ms == 0 {
        return WindowId::time_range(ts_ms, ts_ms);
    }
    let start = (ts_ms / size_ms) * size_ms;
    WindowId::time_range(start, start + size_ms)
}

/// Computes which sliding windows a given timestamp belongs to.
pub fn sliding_windows_for(ts_ms: i64, size: Duration, slide: Duration) -> Vec<WindowId> {
    let size_ms = size.as_millis() as i64;
    let slide_ms = slide.as_millis() as i64;
    if slide_ms == 0 || size_ms == 0 {
        return Vec::new();
    }
    let latest_start = (ts_ms / slide_ms) * slide_ms;
    let earliest_start = latest_start - size_ms + slide_ms;

    let mut windows = Vec::new();
    let mut start = earliest_start;
    while start <= latest_start {
        let end = start + size_ms;
        if ts_ms >= start && ts_ms < end {
            windows.push(WindowId::time_range(start, end));
        }
        start += slide_ms;
    }
    windows
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create events at given timestamps
    fn events(timestamps: &[i64]) -> Vec<WindowEvent<i64>> {
        timestamps
            .iter()
            .map(|&ts| WindowEvent::new(ts, ts))
            .collect()
    }

    fn event_at(ts: i64) -> WindowEvent<i64> {
        WindowEvent::new(ts, ts)
    }

    // ═══ WindowId tests ═══════════════════════════════════

    #[test]
    fn test_window_id_time_range() {
        let wid = WindowId::time_range(1000, 2000);
        assert_eq!(wid.start, 1000);
        assert_eq!(wid.end, 2000);
        assert!(wid.key.is_none());
    }

    #[test]
    fn test_window_id_keyed() {
        let wid = WindowId::keyed(0, 100, "sensor-1");
        assert_eq!(wid.key, Some("sensor-1".to_string()));
    }

    #[test]
    fn test_window_id_duration() {
        let wid = WindowId::time_range(1000, 2000);
        assert_eq!(wid.duration_ms(), 1000);
    }

    #[test]
    fn test_window_id_contains() {
        let wid = WindowId::time_range(1000, 2000);
        assert!(wid.contains(1000));
        assert!(wid.contains(1500));
        assert!(wid.contains(1999));
        assert!(!wid.contains(2000)); // exclusive end
        assert!(!wid.contains(999));
    }

    // ═══ WindowEvent tests ═══════════════════════════════

    #[test]
    fn test_window_event_creation() {
        let evt = WindowEvent::new(42, 1000);
        assert_eq!(evt.value, 42);
        assert_eq!(evt.timestamp_ms, 1000);
    }

    // ═══ WindowPane tests ═══════════════════════════════

    #[test]
    fn test_window_pane_empty() {
        let pane = WindowPane::<i64>::new(WindowId::time_range(0, 1000));
        assert!(pane.is_empty());
        assert_eq!(pane.len(), 0);
        assert_eq!(pane.state, WindowState::Open);
    }

    #[test]
    fn test_window_pane_min_max_timestamp() {
        let mut pane = WindowPane::<i64>::new(WindowId::time_range(0, 1000));
        pane.events.push(WindowEvent::new(1, 500));
        pane.events.push(WindowEvent::new(2, 100));
        pane.events.push(WindowEvent::new(3, 900));
        assert_eq!(pane.min_timestamp(), Some(100));
        assert_eq!(pane.max_timestamp(), Some(900));
    }

    #[test]
    fn test_window_pane_no_timestamps() {
        let pane = WindowPane::<i64>::new(WindowId::time_range(0, 1000));
        assert_eq!(pane.min_timestamp(), None);
        assert_eq!(pane.max_timestamp(), None);
    }

    // ═══ Tumbling window tests ═══════════════════════════

    #[test]
    fn test_tumbling_basic() {
        let mut wa = WindowAlgebra::<i64>::tumbling(Duration::from_secs(10));
        // Events at 0, 5, 9 should all be in [0, 10000)
        for ts in [0, 5000, 9999] {
            wa.ingest(event_at(ts));
        }
        assert_eq!(wa.stats().total_events, 3);
        assert_eq!(wa.open_pane_count(), 1);
    }

    #[test]
    fn test_tumbling_multiple_windows() {
        let mut wa = WindowAlgebra::<i64>::tumbling(Duration::from_secs(10));
        // Window [0..10000)
        wa.ingest(event_at(0));
        wa.ingest(event_at(5000));
        // Window [10000..20000)
        wa.ingest(event_at(10000));
        wa.ingest(event_at(15000));
        assert_eq!(wa.open_pane_count(), 2);
    }

    #[test]
    fn test_tumbling_watermark_closes_window() {
        let mut wa = WindowAlgebra::<i64>::tumbling(Duration::from_secs(10));
        wa.ingest(event_at(0));
        wa.ingest(event_at(5000));

        let outputs = wa.advance_watermark(10000);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].window_id.start, 0);
        assert_eq!(outputs[0].window_id.end, 10000);
        assert_eq!(outputs[0].events.len(), 2);
        assert!(!outputs[0].is_partial);
    }

    #[test]
    fn test_tumbling_watermark_no_close_if_below() {
        let mut wa = WindowAlgebra::<i64>::tumbling(Duration::from_secs(10));
        wa.ingest(event_at(0));
        let outputs = wa.advance_watermark(5000);
        assert!(outputs.is_empty());
    }

    #[test]
    fn test_tumbling_late_event_dropped() {
        let mut wa = WindowAlgebra::<i64>::tumbling(Duration::from_secs(10));
        wa.advance_watermark(20000);
        wa.ingest(event_at(5000)); // late
        assert_eq!(wa.stats().late_events_dropped, 1);
    }

    #[test]
    fn test_tumbling_late_event_side_output() {
        let mut wa = WindowAlgebra::<i64>::tumbling(Duration::from_secs(10))
            .with_late_policy(LatePolicy::SideOutput);
        wa.advance_watermark(20000);
        wa.ingest(event_at(5000)); // late
        assert_eq!(wa.stats().late_events_side_output, 1);
        let side = wa.drain_side_output();
        assert_eq!(side.len(), 1);
        assert_eq!(side[0].timestamp_ms, 5000);
    }

    #[test]
    fn test_tumbling_flush() {
        let mut wa = WindowAlgebra::<i64>::tumbling(Duration::from_secs(10));
        wa.ingest(event_at(0));
        wa.ingest(event_at(5000));
        wa.ingest(event_at(15000));

        let outputs = wa.flush();
        assert_eq!(outputs.len(), 2);
        assert!(outputs.iter().all(|o| o.is_partial));
    }

    // ═══ Sliding window tests ════════════════════════════

    #[test]
    fn test_sliding_basic() {
        let mut wa = WindowAlgebra::<i64>::sliding(Duration::from_secs(10), Duration::from_secs(5));
        wa.ingest(event_at(7500));
        // Event at 7500 with size=10s, slide=5s should be in windows starting at 0 and 5000
        assert!(wa.open_pane_count() >= 1);
    }

    #[test]
    fn test_sliding_event_in_multiple_windows() {
        let mut wa = WindowAlgebra::<i64>::sliding(Duration::from_secs(10), Duration::from_secs(5));
        wa.ingest(event_at(6000));
        // Event at 6000ms with size=10000ms, slide=5000ms:
        // Window [0, 10000) contains it, Window [5000, 15000) contains it
        assert_eq!(wa.open_pane_count(), 2);
    }

    #[test]
    fn test_sliding_watermark_closes_old_windows() {
        let mut wa = WindowAlgebra::<i64>::sliding(Duration::from_secs(10), Duration::from_secs(5));
        wa.ingest(event_at(3000));
        wa.ingest(event_at(6000));
        let outputs = wa.advance_watermark(10000);
        // Window [0, 10000) should be closed
        assert!(!outputs.is_empty());
    }

    #[test]
    fn test_sliding_window_helper() {
        let windows = sliding_windows_for(7500, Duration::from_secs(10), Duration::from_secs(5));
        assert!(!windows.is_empty());
        assert!(windows.iter().all(|w| w.contains(7500)));
    }

    // ═══ Session window tests ════════════════════════════

    #[test]
    fn test_session_basic() {
        let mut wa = WindowAlgebra::<i64>::session(Duration::from_secs(5));
        wa.ingest(event_at(1000));
        wa.ingest(event_at(3000));
        wa.ingest(event_at(4000));
        // All within 5s gap, should be one session
        assert_eq!(wa.open_pane_count(), 1);
    }

    #[test]
    fn test_session_gap_closes_window() {
        let mut wa = WindowAlgebra::<i64>::session(Duration::from_secs(5));
        wa.ingest(event_at(1000));
        wa.ingest(event_at(3000));
        // Gap > 5000ms, should close previous session
        let outputs = wa.ingest(event_at(10000));
        assert!(!outputs.is_empty());
    }

    #[test]
    fn test_session_multiple_sessions() {
        let mut wa = WindowAlgebra::<i64>::session(Duration::from_secs(5));
        wa.ingest(event_at(1000));
        wa.ingest(event_at(3000));
        let out1 = wa.ingest(event_at(20000)); // gap > 5s
                                               // First session closed
        assert!(!out1.is_empty());
        wa.ingest(event_at(22000));
        let out2 = wa.ingest(event_at(40000)); // gap > 5s
        assert!(!out2.is_empty());
    }

    // ═══ Count window tests ══════════════════════════════

    #[test]
    fn test_count_basic() {
        let mut wa = WindowAlgebra::<i64>::count(3);
        let out1 = wa.ingest(event_at(1));
        assert!(out1.is_empty());
        let out2 = wa.ingest(event_at(2));
        assert!(out2.is_empty());
        let out3 = wa.ingest(event_at(3));
        assert_eq!(out3.len(), 1);
        assert_eq!(out3[0].events.len(), 3);
    }

    #[test]
    fn test_count_multiple_triggers() {
        let mut wa = WindowAlgebra::<i64>::count(2);
        let evts = events(&[1, 2, 3, 4, 5, 6]);
        let outputs = wa.ingest_batch(evts);
        assert_eq!(outputs.len(), 3);
    }

    #[test]
    fn test_count_partial_flush() {
        let mut wa = WindowAlgebra::<i64>::count(3);
        wa.ingest(event_at(1));
        wa.ingest(event_at(2));
        let outputs = wa.flush();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_partial);
        assert_eq!(outputs[0].events.len(), 2);
    }

    // ═══ Late data policy tests ══════════════════════════

    #[test]
    fn test_late_policy_drop() {
        let mut wa = WindowAlgebra::<i64>::tumbling(Duration::from_secs(10))
            .with_late_policy(LatePolicy::Drop);
        wa.advance_watermark(30000);
        wa.ingest(event_at(5000));
        assert_eq!(wa.stats().late_events_dropped, 1);
        assert_eq!(wa.drain_side_output().len(), 0);
    }

    #[test]
    fn test_late_policy_side_output() {
        let mut wa = WindowAlgebra::<i64>::tumbling(Duration::from_secs(10))
            .with_late_policy(LatePolicy::SideOutput);
        wa.advance_watermark(30000);
        wa.ingest(event_at(5000));
        wa.ingest(event_at(8000));
        assert_eq!(wa.stats().late_events_side_output, 2);
        let side = wa.drain_side_output();
        assert_eq!(side.len(), 2);
    }

    #[test]
    fn test_late_policy_allowed_lateness() {
        let mut wa = WindowAlgebra::<i64>::tumbling(Duration::from_secs(10)).with_late_policy(
            LatePolicy::AllowedLateness {
                lateness: Duration::from_secs(5),
            },
        );
        // Create window [0..10000), event in it
        wa.ingest(event_at(5000));
        // Advance watermark just past window end but within lateness
        // Events before watermark that have no existing window -> late
        wa.advance_watermark(30000);
        wa.ingest(event_at(2000)); // late, past allowed lateness too
        assert_eq!(wa.stats().late_events_dropped, 1);
    }

    // ═══ Max open windows / eviction tests ═══════════════

    #[test]
    fn test_max_open_windows_eviction() {
        let mut wa =
            WindowAlgebra::<i64>::tumbling(Duration::from_secs(1)).with_max_open_windows(3);
        // Create 4 windows
        wa.ingest(event_at(0));
        wa.ingest(event_at(1000));
        wa.ingest(event_at(2000));
        let outputs = wa.ingest(event_at(3000));
        // Should evict the oldest
        assert!(wa.stats().windows_evicted >= 1 || !outputs.is_empty());
    }

    #[test]
    fn test_evicted_window_emits_partial() {
        let mut wa =
            WindowAlgebra::<i64>::tumbling(Duration::from_secs(1)).with_max_open_windows(2);
        wa.ingest(event_at(0));
        wa.ingest(event_at(1000));
        let outputs = wa.ingest(event_at(2000));
        let partial = outputs.iter().filter(|o| o.is_partial).count();
        assert!(partial >= 1);
    }

    // ═══ Statistics tests ════════════════════════════════

    #[test]
    fn test_stats_total_events() {
        let mut wa = WindowAlgebra::<i64>::tumbling(Duration::from_secs(10));
        for i in 0..10 {
            wa.ingest(event_at(i * 100));
        }
        assert_eq!(wa.stats().total_events, 10);
    }

    #[test]
    fn test_stats_windows_opened() {
        let mut wa = WindowAlgebra::<i64>::tumbling(Duration::from_secs(10));
        wa.ingest(event_at(0));
        wa.ingest(event_at(10000));
        wa.ingest(event_at(20000));
        assert_eq!(wa.stats().windows_opened, 3);
    }

    #[test]
    fn test_stats_windows_closed() {
        let mut wa = WindowAlgebra::<i64>::tumbling(Duration::from_secs(10));
        wa.ingest(event_at(0));
        wa.ingest(event_at(10000));
        wa.advance_watermark(20000);
        assert_eq!(wa.stats().windows_closed, 2);
    }

    // ═══ Aggregation tests ═══════════════════════════════

    #[test]
    fn test_aggregate_sum() {
        let mut wa = WindowAlgebra::<i64>::tumbling(Duration::from_secs(10));
        wa.ingest(event_at(100));
        wa.ingest(event_at(200));
        wa.ingest(event_at(300));

        let wid = WindowId::time_range(0, 10000);
        let sum = wa.aggregate(&wid, 0i64, |acc, &val| acc + val);
        assert_eq!(sum, Some(600));
    }

    #[test]
    fn test_aggregate_count() {
        let mut wa = WindowAlgebra::<i64>::tumbling(Duration::from_secs(10));
        wa.ingest(event_at(100));
        wa.ingest(event_at(200));

        let wid = WindowId::time_range(0, 10000);
        let count = wa.aggregate(&wid, 0usize, |acc, _| acc + 1);
        assert_eq!(count, Some(2));
    }

    #[test]
    fn test_aggregate_nonexistent_window() {
        let wa = WindowAlgebra::<i64>::tumbling(Duration::from_secs(10));
        let wid = WindowId::time_range(0, 10000);
        let result = wa.aggregate(&wid, 0, |acc, _: &i64| acc + 1);
        assert!(result.is_none());
    }

    // ═══ Helper function tests ═══════════════════════════

    #[test]
    fn test_tumbling_window_for_helper() {
        let wid = tumbling_window_for(7500, Duration::from_secs(10));
        assert_eq!(wid.start, 0);
        assert_eq!(wid.end, 10000);
    }

    #[test]
    fn test_tumbling_window_for_exact_boundary() {
        let wid = tumbling_window_for(10000, Duration::from_secs(10));
        assert_eq!(wid.start, 10000);
        assert_eq!(wid.end, 20000);
    }

    #[test]
    fn test_sliding_windows_for_helper() {
        let windows = sliding_windows_for(12000, Duration::from_secs(10), Duration::from_secs(5));
        // 12000 should be in [5000, 15000) and [10000, 20000)
        assert!(!windows.is_empty());
        for w in &windows {
            assert!(w.contains(12000));
        }
    }

    // ═══ Batch ingest tests ══════════════════════════════

    #[test]
    fn test_batch_ingest() {
        let mut wa = WindowAlgebra::<i64>::count(3);
        let evts = events(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let outputs = wa.ingest_batch(evts);
        assert_eq!(outputs.len(), 3);
        assert!(outputs.iter().all(|o| o.events.len() == 3));
    }

    // ═══ WindowAlgebraConfig tests ═══════════════════════

    #[test]
    fn test_default_config() {
        let config = WindowAlgebraConfig::default();
        assert_eq!(
            config.kind,
            WindowKind::Tumbling {
                size: Duration::from_secs(60)
            }
        );
        assert_eq!(config.late_policy, LatePolicy::Drop);
        assert_eq!(config.max_open_windows, 10_000);
        assert!(config.emit_on_evict);
    }

    #[test]
    fn test_custom_config() {
        let config = WindowAlgebraConfig {
            kind: WindowKind::Sliding {
                size: Duration::from_secs(30),
                slide: Duration::from_secs(10),
            },
            late_policy: LatePolicy::SideOutput,
            max_open_windows: 500,
            emit_on_evict: false,
        };
        assert_eq!(config.max_open_windows, 500);
        assert!(!config.emit_on_evict);
    }

    // ═══ WindowState tests ═══════════════════════════════

    #[test]
    fn test_window_state_variants() {
        assert_eq!(WindowState::Open, WindowState::Open);
        assert_ne!(WindowState::Open, WindowState::Closed);
        assert_ne!(WindowState::Closing, WindowState::Closed);
    }

    // ═══ Edge case tests ═════════════════════════════════

    #[test]
    fn test_watermark_no_regression() {
        let mut wa = WindowAlgebra::<i64>::tumbling(Duration::from_secs(10));
        wa.advance_watermark(10000);
        let outputs = wa.advance_watermark(5000); // regression
        assert!(outputs.is_empty());
        assert_eq!(wa.watermark_ms(), 10000);
    }

    #[test]
    fn test_empty_flush() {
        let mut wa = WindowAlgebra::<i64>::tumbling(Duration::from_secs(10));
        let outputs = wa.flush();
        assert!(outputs.is_empty());
    }

    #[test]
    fn test_double_watermark_advance() {
        let mut wa = WindowAlgebra::<i64>::tumbling(Duration::from_secs(10));
        wa.ingest(event_at(5000));
        let out1 = wa.advance_watermark(10000);
        assert_eq!(out1.len(), 1);
        let out2 = wa.advance_watermark(10000);
        assert!(out2.is_empty()); // already closed
    }
}
