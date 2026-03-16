//! CAN frame temporal aggregation (count, min, max, avg per window).
//!
//! `FrameAggregator` collects incoming CAN frames, maintains per-ID statistics,
//! and exposes a summary of all frames within the configured time window.

use std::collections::{HashMap, HashSet, VecDeque};

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// A single CAN frame with a 29-bit or 11-bit ID and a payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanFrame {
    /// CAN frame identifier.
    pub id: u32,
    /// Payload bytes (up to 8 for CAN 2.0, up to 64 for CAN FD).
    pub data: Vec<u8>,
    /// Epoch-millis timestamp when the frame was captured.
    pub timestamp_ms: u64,
}

impl CanFrame {
    /// Construct a new CAN frame.
    pub fn new(id: u32, data: Vec<u8>, timestamp_ms: u64) -> Self {
        Self {
            id,
            data,
            timestamp_ms,
        }
    }
}

/// Summary of all frames within a time window.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregationWindow {
    /// Window start (epoch millis, inclusive).
    pub start_ms: u64,
    /// Window end (epoch millis, exclusive).
    pub end_ms: u64,
    /// Total number of frames in the window.
    pub frame_count: usize,
    /// Unique frame IDs seen in the window.
    pub ids_seen: HashSet<u32>,
    /// Total payload bytes across all frames in the window.
    pub bytes_total: usize,
}

/// Per-ID statistics over all frames received since tracking began.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameStats {
    /// CAN frame ID.
    pub frame_id: u32,
    /// Total frames received with this ID.
    pub count: usize,
    /// Epoch-millis of the first frame with this ID.
    pub first_ms: u64,
    /// Epoch-millis of the most recent frame with this ID.
    pub last_ms: u64,
    /// Average milliseconds between consecutive frames with this ID.
    pub avg_interval_ms: f64,
    /// Total payload bytes across all frames with this ID.
    pub data_bytes_total: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// FrameAggregator
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregates CAN frames over a sliding time window.
///
/// Internal storage:
/// - A `VecDeque<CanFrame>` for fast O(1) eviction at the front.
/// - A `HashMap<u32, FrameStats>` for per-ID running statistics.
///
/// Statistics accumulate indefinitely across evictions; the window summary
/// only covers frames still in the deque.
#[derive(Debug)]
pub struct FrameAggregator {
    /// Width of the aggregation window in milliseconds.
    window_ms: u64,
    /// Frames currently in the window, oldest at the front.
    frames: VecDeque<CanFrame>,
    /// Running statistics per frame ID.
    stats: HashMap<u32, FrameStats>,
}

impl FrameAggregator {
    /// Create a new aggregator with the given window width.
    pub fn new(window_ms: u64) -> Self {
        Self {
            window_ms,
            frames: VecDeque::new(),
            stats: HashMap::new(),
        }
    }

    /// Push a new frame and update per-ID statistics.
    ///
    /// Old frames (those whose timestamp falls outside the window ending at
    /// `frame.timestamp_ms`) are evicted automatically.
    pub fn push(&mut self, frame: CanFrame) {
        // Evict frames that have fallen outside the window.
        let cutoff = frame.timestamp_ms.saturating_sub(self.window_ms);
        self.evict_before(cutoff);

        // Update per-ID running statistics.
        let entry = self.stats.entry(frame.id).or_insert_with(|| FrameStats {
            frame_id: frame.id,
            count: 0,
            first_ms: frame.timestamp_ms,
            last_ms: frame.timestamp_ms,
            avg_interval_ms: 0.0,
            data_bytes_total: 0,
        });

        if entry.count > 0 {
            let new_interval = frame.timestamp_ms.saturating_sub(entry.last_ms) as f64;
            // Incremental mean: avg_new = avg_old + (x_new − avg_old) / n
            entry.avg_interval_ms += (new_interval - entry.avg_interval_ms) / entry.count as f64;
        }

        entry.last_ms = frame.timestamp_ms;
        entry.data_bytes_total += frame.data.len();
        entry.count += 1;

        self.frames.push_back(frame);
    }

    /// Return a summary of all frames currently within the window.
    pub fn window_summary(&self, now_ms: u64) -> AggregationWindow {
        let end_ms = now_ms;
        let start_ms = end_ms.saturating_sub(self.window_ms);

        let mut ids_seen = HashSet::new();
        let mut bytes_total = 0_usize;
        let mut frame_count = 0_usize;

        for frame in &self.frames {
            if frame.timestamp_ms >= start_ms {
                ids_seen.insert(frame.id);
                bytes_total += frame.data.len();
                frame_count += 1;
            }
        }

        AggregationWindow {
            start_ms,
            end_ms,
            frame_count,
            ids_seen,
            bytes_total,
        }
    }

    /// Return statistics for a specific frame ID, if any frames with that ID
    /// have been pushed.
    pub fn stats_for(&self, frame_id: u32) -> Option<&FrameStats> {
        self.stats.get(&frame_id)
    }

    /// Return statistics for all tracked frame IDs.
    pub fn all_stats(&self) -> Vec<&FrameStats> {
        self.stats.values().collect()
    }

    /// Remove all frames with `timestamp_ms < cutoff_ms` from the internal
    /// deque.  Returns the number of frames evicted.
    pub fn evict_before(&mut self, cutoff_ms: u64) -> usize {
        let mut count = 0_usize;
        while let Some(front) = self.frames.front() {
            if front.timestamp_ms < cutoff_ms {
                self.frames.pop_front();
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    /// Return the total number of frames currently in the deque (within window).
    pub fn total_frames(&self) -> usize {
        self.frames.len()
    }

    /// Return the number of unique frame IDs for which statistics exist.
    pub fn unique_ids(&self) -> usize {
        self.stats.len()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(id: u32, data: Vec<u8>, ts: u64) -> CanFrame {
        CanFrame::new(id, data, ts)
    }

    fn simple_frame(id: u32, ts: u64) -> CanFrame {
        frame(id, vec![0xAA, 0xBB], ts)
    }

    // ── CanFrame ──────────────────────────────────────────────────────────────

    #[test]
    fn test_can_frame_new() {
        let f = CanFrame::new(0x1A2, vec![1, 2, 3], 12345);
        assert_eq!(f.id, 0x1A2);
        assert_eq!(f.data, vec![1, 2, 3]);
        assert_eq!(f.timestamp_ms, 12345);
    }

    #[test]
    fn test_can_frame_equality() {
        let f1 = simple_frame(1, 100);
        let f2 = simple_frame(1, 100);
        assert_eq!(f1, f2);
    }

    #[test]
    fn test_can_frame_clone() {
        let f = simple_frame(5, 500);
        let f2 = f.clone();
        assert_eq!(f, f2);
    }

    // ── FrameAggregator construction ──────────────────────────────────────────

    #[test]
    fn test_new_is_empty() {
        let agg = FrameAggregator::new(1000);
        assert_eq!(agg.total_frames(), 0);
        assert_eq!(agg.unique_ids(), 0);
    }

    #[test]
    fn test_push_one_frame() {
        let mut agg = FrameAggregator::new(5000);
        agg.push(simple_frame(0x100, 1000));
        assert_eq!(agg.total_frames(), 1);
        assert_eq!(agg.unique_ids(), 1);
    }

    #[test]
    fn test_push_multiple_frames_same_id() {
        let mut agg = FrameAggregator::new(10000);
        for i in 0..5_u64 {
            agg.push(simple_frame(0x200, i * 100));
        }
        assert_eq!(agg.total_frames(), 5);
        assert_eq!(agg.unique_ids(), 1);
    }

    #[test]
    fn test_push_multiple_ids() {
        let mut agg = FrameAggregator::new(10000);
        agg.push(simple_frame(0x1, 1000));
        agg.push(simple_frame(0x2, 2000));
        agg.push(simple_frame(0x3, 3000));
        assert_eq!(agg.unique_ids(), 3);
    }

    // ── Eviction ──────────────────────────────────────────────────────────────

    #[test]
    fn test_evict_before_removes_old_frames() {
        let mut agg = FrameAggregator::new(5000);
        agg.push(simple_frame(1, 100));
        agg.push(simple_frame(2, 200));
        agg.push(simple_frame(3, 300));

        let evicted = agg.evict_before(250);
        assert_eq!(evicted, 2); // ts 100 and ts 200 are < 250
        assert_eq!(agg.total_frames(), 1);
    }

    #[test]
    fn test_evict_before_zero_removes_nothing() {
        let mut agg = FrameAggregator::new(5000);
        agg.push(simple_frame(1, 100));
        let evicted = agg.evict_before(0);
        assert_eq!(evicted, 0);
        assert_eq!(agg.total_frames(), 1);
    }

    #[test]
    fn test_auto_eviction_on_push() {
        let mut agg = FrameAggregator::new(500); // 500 ms window
        agg.push(simple_frame(1, 0));
        agg.push(simple_frame(2, 300));
        // Push a frame at 600 ms → frames at ts 0 (< 600-500=100) should evict
        agg.push(simple_frame(3, 600));
        // Frame at ts 0 is evicted; frames at ts 300 and 600 remain
        assert_eq!(agg.total_frames(), 2);
    }

    #[test]
    fn test_evict_all_frames() {
        let mut agg = FrameAggregator::new(5000);
        agg.push(simple_frame(1, 100));
        agg.push(simple_frame(2, 200));
        let evicted = agg.evict_before(1000);
        assert_eq!(evicted, 2);
        assert_eq!(agg.total_frames(), 0);
    }

    // ── window_summary ────────────────────────────────────────────────────────

    #[test]
    fn test_window_summary_empty() {
        let agg = FrameAggregator::new(1000);
        let summary = agg.window_summary(5000);
        assert_eq!(summary.frame_count, 0);
        assert!(summary.ids_seen.is_empty());
        assert_eq!(summary.bytes_total, 0);
    }

    #[test]
    fn test_window_summary_frame_count() {
        let mut agg = FrameAggregator::new(1000);
        agg.push(simple_frame(1, 4000));
        agg.push(simple_frame(2, 4500));
        let summary = agg.window_summary(5000);
        assert_eq!(summary.frame_count, 2);
    }

    #[test]
    fn test_window_summary_bytes_total() {
        let mut agg = FrameAggregator::new(1000);
        agg.push(frame(1, vec![1, 2, 3, 4], 4500)); // 4 bytes
        agg.push(frame(2, vec![5, 6], 4800)); // 2 bytes
        let summary = agg.window_summary(5000);
        assert_eq!(summary.bytes_total, 6);
    }

    #[test]
    fn test_window_summary_ids_seen() {
        let mut agg = FrameAggregator::new(1000);
        agg.push(simple_frame(0xA, 4500));
        agg.push(simple_frame(0xB, 4700));
        agg.push(simple_frame(0xA, 4900));
        let summary = agg.window_summary(5000);
        assert!(summary.ids_seen.contains(&0xA));
        assert!(summary.ids_seen.contains(&0xB));
        assert_eq!(summary.ids_seen.len(), 2);
    }

    #[test]
    fn test_window_summary_start_end() {
        let agg = FrameAggregator::new(500);
        let summary = agg.window_summary(1000);
        assert_eq!(summary.start_ms, 500);
        assert_eq!(summary.end_ms, 1000);
    }

    // ── stats_for / all_stats ─────────────────────────────────────────────────

    #[test]
    fn test_stats_for_unknown_id_is_none() {
        let agg = FrameAggregator::new(1000);
        assert!(agg.stats_for(0xFF).is_none());
    }

    #[test]
    fn test_stats_for_known_id() {
        let mut agg = FrameAggregator::new(1000);
        agg.push(simple_frame(0x10, 100));
        let stats = agg.stats_for(0x10).expect("stats should exist");
        assert_eq!(stats.frame_id, 0x10);
        assert_eq!(stats.count, 1);
    }

    #[test]
    fn test_stats_count_increments() {
        let mut agg = FrameAggregator::new(10000);
        for ts in [100_u64, 200, 300, 400, 500] {
            agg.push(simple_frame(0x20, ts));
        }
        let stats = agg.stats_for(0x20).expect("should succeed");
        assert_eq!(stats.count, 5);
    }

    #[test]
    fn test_stats_first_and_last_ms() {
        let mut agg = FrameAggregator::new(10000);
        agg.push(simple_frame(0x30, 1000));
        agg.push(simple_frame(0x30, 2000));
        agg.push(simple_frame(0x30, 3000));
        let stats = agg.stats_for(0x30).expect("should succeed");
        assert_eq!(stats.first_ms, 1000);
        assert_eq!(stats.last_ms, 3000);
    }

    #[test]
    fn test_stats_data_bytes_total() {
        let mut agg = FrameAggregator::new(10000);
        agg.push(frame(0x40, vec![0, 1, 2], 100));
        agg.push(frame(0x40, vec![3, 4], 200));
        let stats = agg.stats_for(0x40).expect("should succeed");
        assert_eq!(stats.data_bytes_total, 5);
    }

    #[test]
    fn test_stats_avg_interval() {
        let mut agg = FrameAggregator::new(10000);
        // Three frames at 0, 100, 200 → intervals are 100 and 100 → avg = 100
        agg.push(simple_frame(0x50, 0));
        agg.push(simple_frame(0x50, 100));
        agg.push(simple_frame(0x50, 200));
        let stats = agg.stats_for(0x50).expect("should succeed");
        assert!((stats.avg_interval_ms - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_all_stats_returns_all_ids() {
        let mut agg = FrameAggregator::new(10000);
        agg.push(simple_frame(1, 100));
        agg.push(simple_frame(2, 200));
        agg.push(simple_frame(3, 300));
        assert_eq!(agg.all_stats().len(), 3);
    }

    // ── unique_ids ────────────────────────────────────────────────────────────

    #[test]
    fn test_unique_ids_zero_initially() {
        let agg = FrameAggregator::new(1000);
        assert_eq!(agg.unique_ids(), 0);
    }

    #[test]
    fn test_unique_ids_after_push() {
        let mut agg = FrameAggregator::new(1000);
        agg.push(simple_frame(0xAA, 1000));
        agg.push(simple_frame(0xBB, 1001));
        agg.push(simple_frame(0xAA, 1002));
        assert_eq!(agg.unique_ids(), 2);
    }

    // ── stats survive eviction ────────────────────────────────────────────────

    #[test]
    fn test_stats_persist_after_eviction() {
        let mut agg = FrameAggregator::new(100);
        agg.push(simple_frame(0x77, 0));
        agg.push(simple_frame(0x77, 1000)); // evicts frame at ts=0
                                            // Stats should still reflect both pushes
        let stats = agg.stats_for(0x77).expect("should succeed");
        assert_eq!(stats.count, 2);
        assert_eq!(agg.total_frames(), 1); // only second frame remains
    }

    // ── edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_empty_payload_frame() {
        let mut agg = FrameAggregator::new(1000);
        agg.push(frame(0x1, vec![], 100));
        let stats = agg.stats_for(0x1).expect("should succeed");
        assert_eq!(stats.data_bytes_total, 0);
    }

    #[test]
    fn test_frame_with_max_id() {
        let mut agg = FrameAggregator::new(1000);
        agg.push(simple_frame(0x1FFFFFFF, 100));
        assert_eq!(agg.unique_ids(), 1);
    }

    #[test]
    fn test_window_zero_width() {
        let mut agg = FrameAggregator::new(0);
        agg.push(simple_frame(1, 1000));
        // Window of 0 ms — no historical frames; only current ts qualifies (>=start)
        let summary = agg.window_summary(1000);
        // start = 1000, end = 1000, frame at ts=1000 qualifies (>= 1000)
        assert_eq!(summary.frame_count, 1);
    }

    #[test]
    fn test_many_frames_performance() {
        let mut agg = FrameAggregator::new(10000);
        for i in 0..1000_u64 {
            agg.push(simple_frame((i % 10) as u32, i * 10));
        }
        assert!(agg.total_frames() > 0);
        assert_eq!(agg.unique_ids(), 10);
    }
}
