//! Tumbling × Sliding watermark-driven join.
//!
//! The *left* stream uses tumbling (fixed, non-overlapping) windows of size
//! `left_size_ms`.  The *right* stream uses sliding windows defined by
//! `right_size_ms` (window duration) and `right_slide_ms` (slide step).
//!
//! Two events join when:
//!
//! * They share the join key, and
//! * The right event's timestamp falls inside any sliding window pane that
//!   overlaps with the left event's tumbling pane.
//!
//! Concretely, an arriving event on either side is matched against every
//! buffered event of the opposite side whose timestamp lies inside one of
//! the relevant sliding panes.
//!
//! Pane closure is watermark driven:
//!
//! * Left tumbling pane `[s, s+sizeL)` closes when
//!   `wm ≥ s + sizeL + allowed_lateness`.
//! * Right sliding pane `[s, s+sizeR)` closes when
//!   `wm ≥ s + sizeR + allowed_lateness`.

use std::collections::HashMap;

use super::{WindowJoinKey, WindowJoinResult, WindowJoinStats};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Tumbling × Sliding join configuration.
#[derive(Debug, Clone)]
pub struct TumblingSlidingJoinConfig {
    /// Tumbling window size for the left side (ms).
    pub left_size_ms: i64,
    /// Sliding window size for the right side (ms).
    pub right_size_ms: i64,
    /// Sliding window slide step for the right side (ms).
    pub right_slide_ms: i64,
    /// Allowed lateness past pane end (ms).
    pub allowed_lateness_ms: i64,
}

impl TumblingSlidingJoinConfig {
    /// Create a config with the supplied parameters and zero lateness.
    pub fn new(left_size_ms: i64, right_size_ms: i64, right_slide_ms: i64) -> Self {
        assert!(left_size_ms > 0, "left_size_ms must be > 0");
        assert!(right_size_ms > 0, "right_size_ms must be > 0");
        assert!(right_slide_ms > 0, "right_slide_ms must be > 0");
        Self {
            left_size_ms,
            right_size_ms,
            right_slide_ms,
            allowed_lateness_ms: 0,
        }
    }

    /// Override allowed lateness.
    pub fn with_lateness(mut self, allowed_lateness_ms: i64) -> Self {
        self.allowed_lateness_ms = allowed_lateness_ms;
        self
    }
}

// ─── Internal state ──────────────────────────────────────────────────────────

#[derive(Clone)]
struct StampedEvent<E: Clone> {
    ts_ms: i64,
    event: E,
}

// ─── TumblingSlidingJoin ─────────────────────────────────────────────────────

/// Watermark-driven tumbling × sliding join.
///
/// Buffers right-side events keyed by `(WindowJoinKey, ts_ms)` so each event
/// is materialised once, regardless of how many overlapping sliding panes
/// it inhabits.  Pane bookkeeping is used solely for purging.
pub struct TumblingSlidingJoin<L: Clone, R: Clone> {
    config: TumblingSlidingJoinConfig,
    // Left buffer keyed by tumbling pane start; for each pane a key→events map.
    left: HashMap<i64, HashMap<WindowJoinKey, Vec<StampedEvent<L>>>>,
    // Right buffer is per sliding pane start; the *same* event may appear in
    // multiple panes (it shares one materialisation but is referenced N times).
    right: HashMap<i64, HashMap<WindowJoinKey, Vec<StampedEvent<R>>>>,
    last_watermark_ms: i64,
    stats: WindowJoinStats,
}

impl<L: Clone, R: Clone> TumblingSlidingJoin<L, R> {
    /// Create a new join.
    pub fn new(config: TumblingSlidingJoinConfig) -> Self {
        Self {
            config,
            left: HashMap::new(),
            right: HashMap::new(),
            last_watermark_ms: i64::MIN,
            stats: WindowJoinStats::default(),
        }
    }

    fn left_pane(&self, ts_ms: i64) -> i64 {
        let s = self.config.left_size_ms;
        ts_ms.div_euclid(s) * s
    }

    /// Compute every sliding pane that contains `ts_ms` for the right side.
    fn right_panes_for(&self, ts_ms: i64) -> Vec<i64> {
        let size = self.config.right_size_ms;
        let slide = self.config.right_slide_ms;
        // Right pane starts s satisfy: s ≤ ts < s + size, i.e.
        //     s ≤ ts and s > ts - size
        // and s = k * slide for k integer.
        // ⇒  ts - size < s ≤ ts and s % slide == 0.
        let lower_excl = ts_ms - size;
        let upper_incl = ts_ms;
        // smallest k with k*slide > lower_excl
        let lower_k_floor = lower_excl.div_euclid(slide);
        let mut start_k = lower_k_floor + 1;
        // ensure k*slide > lower_excl strictly
        while start_k * slide <= lower_excl {
            start_k += 1;
        }
        let end_k = upper_incl.div_euclid(slide);
        let mut panes = Vec::new();
        for k in start_k..=end_k {
            let s = k * slide;
            if s > lower_excl && s <= upper_incl {
                panes.push(s);
            }
        }
        panes
    }

    fn left_pane_closed(&self, pane_start: i64) -> bool {
        let close_at = pane_start
            .saturating_add(self.config.left_size_ms)
            .saturating_add(self.config.allowed_lateness_ms);
        self.last_watermark_ms >= close_at
    }

    fn right_pane_closed(&self, pane_start: i64) -> bool {
        let close_at = pane_start
            .saturating_add(self.config.right_size_ms)
            .saturating_add(self.config.allowed_lateness_ms);
        self.last_watermark_ms >= close_at
    }

    /// Insert a left event.
    pub fn push_left(
        &mut self,
        key: WindowJoinKey,
        ts_ms: i64,
        event: L,
    ) -> Vec<WindowJoinResult<L, R>> {
        let pane_start = self.left_pane(ts_ms);
        if self.left_pane_closed(pane_start) {
            self.stats.late_events_dropped += 1;
            return Vec::new();
        }
        self.stats.left_events += 1;
        let pane_end = pane_start + self.config.left_size_ms;

        // Match against every right pane that overlaps this left pane,
        // de-duplicating right events that appear in multiple sliding panes.
        let mut emitted: Vec<WindowJoinResult<L, R>> = Vec::new();
        let mut seen_right_ts: Vec<i64> = Vec::new();
        let right_panes_overlapping = self.right_panes_overlapping_left(pane_start);
        for r_start in right_panes_overlapping {
            if let Some(events_by_key) = self.right.get(&r_start) {
                if let Some(events) = events_by_key.get(&key) {
                    for r_ev in events {
                        if r_ev.ts_ms >= pane_start
                            && r_ev.ts_ms < pane_end
                            && !seen_right_ts.contains(&r_ev.ts_ms)
                        {
                            seen_right_ts.push(r_ev.ts_ms);
                            emitted.push(WindowJoinResult {
                                key: key.clone(),
                                left: event.clone(),
                                right: r_ev.event.clone(),
                                pane_end_ms: pane_end,
                            });
                        }
                    }
                }
            }
        }

        self.left
            .entry(pane_start)
            .or_default()
            .entry(key)
            .or_default()
            .push(StampedEvent { ts_ms, event });
        self.stats.joined_pairs += emitted.len() as u64;
        emitted
    }

    /// Insert a right event.
    pub fn push_right(
        &mut self,
        key: WindowJoinKey,
        ts_ms: i64,
        event: R,
    ) -> Vec<WindowJoinResult<L, R>> {
        let panes = self.right_panes_for(ts_ms);
        // If every pane the event would fall into is closed, drop as late.
        let any_open = panes.iter().any(|&s| !self.right_pane_closed(s));
        if !any_open {
            self.stats.late_events_dropped += 1;
            return Vec::new();
        }
        self.stats.right_events += 1;

        // Match against the left tumbling pane that contains `ts_ms`.
        let left_pane_start = self.left_pane(ts_ms);
        let left_pane_end = left_pane_start + self.config.left_size_ms;
        let mut emitted = Vec::new();
        if let Some(events_by_key) = self.left.get(&left_pane_start) {
            if let Some(events) = events_by_key.get(&key) {
                for l_ev in events {
                    emitted.push(WindowJoinResult {
                        key: key.clone(),
                        left: l_ev.event.clone(),
                        right: event.clone(),
                        pane_end_ms: left_pane_end,
                    });
                }
            }
        }

        // Buffer the event in every open sliding pane.
        for s in panes {
            if !self.right_pane_closed(s) {
                self.right
                    .entry(s)
                    .or_default()
                    .entry(key.clone())
                    .or_default()
                    .push(StampedEvent {
                        ts_ms,
                        event: event.clone(),
                    });
            }
        }

        self.stats.joined_pairs += emitted.len() as u64;
        emitted
    }

    /// All sliding-pane starts whose pane overlaps the given left tumbling pane.
    fn right_panes_overlapping_left(&self, left_pane_start: i64) -> Vec<i64> {
        let l_end = left_pane_start + self.config.left_size_ms;
        let mut out = Vec::new();
        for &r_start in self.right.keys() {
            let r_end = r_start + self.config.right_size_ms;
            if r_start < l_end && r_end > left_pane_start {
                out.push(r_start);
            }
        }
        out
    }

    /// Advance the input watermark.  Returns total panes purged (left + right).
    pub fn advance_watermark(&mut self, watermark_ms: i64) -> usize {
        if watermark_ms < self.last_watermark_ms {
            return 0;
        }
        self.last_watermark_ms = watermark_ms;
        let lat = self.config.allowed_lateness_ms;
        let l_size = self.config.left_size_ms;
        let r_size = self.config.right_size_ms;

        let mut purged = 0usize;
        self.left.retain(|&start, _| {
            let close_at = start.saturating_add(l_size).saturating_add(lat);
            let keep = watermark_ms < close_at;
            if !keep {
                purged += 1;
            }
            keep
        });
        self.right.retain(|&start, _| {
            let close_at = start.saturating_add(r_size).saturating_add(lat);
            let keep = watermark_ms < close_at;
            if !keep {
                purged += 1;
            }
            keep
        });
        self.stats.windows_closed += purged as u64;
        purged
    }

    /// Statistics snapshot.
    pub fn stats(&self) -> &WindowJoinStats {
        &self.stats
    }

    /// Number of currently buffered left/right panes (sum).
    pub fn pane_count(&self) -> usize {
        self.left.len() + self.right.len()
    }

    /// Most recently observed watermark.
    pub fn watermark(&self) -> i64 {
        self.last_watermark_ms
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_when_event_falls_in_tumbling_pane() {
        // left tumbling 1000ms, right sliding 1000/500.
        let cfg = TumblingSlidingJoinConfig::new(1_000, 1_000, 500);
        let mut j: TumblingSlidingJoin<&str, &str> = TumblingSlidingJoin::new(cfg);
        // Left at 200ms in pane [0,1000).
        j.push_left("k".into(), 200, "L0");
        // Right at 600ms is in sliding panes 0 and 500; both overlap left pane.
        let r = j.push_right("k".into(), 600, "R0");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].pane_end_ms, 1_000);
    }

    #[test]
    fn does_not_join_when_outside_tumbling_pane() {
        let cfg = TumblingSlidingJoinConfig::new(1_000, 1_000, 500);
        let mut j: TumblingSlidingJoin<&str, &str> = TumblingSlidingJoin::new(cfg);
        j.push_left("k".into(), 200, "L0");
        // Right at 1100ms — falls in sliding panes 500, 1000 → join only with L
        // events whose tumbling pane contains 1100, i.e. [1000,2000). Empty.
        let r = j.push_right("k".into(), 1_100, "R0");
        assert!(r.is_empty());
    }

    #[test]
    fn left_event_matches_existing_right_in_overlapping_pane() {
        let cfg = TumblingSlidingJoinConfig::new(1_000, 1_000, 500);
        let mut j: TumblingSlidingJoin<&str, &str> = TumblingSlidingJoin::new(cfg);
        // Right first at 300ms → buffers in pane 0 (size 1000) and pane -500.
        let _ = j.push_right("k".into(), 300, "R0");
        // Then left at 200ms — should emit the join.
        let r = j.push_left("k".into(), 200, "L0");
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn watermark_purges_left_and_right_panes() {
        let cfg = TumblingSlidingJoinConfig::new(1_000, 1_000, 500);
        let mut j: TumblingSlidingJoin<&str, &str> = TumblingSlidingJoin::new(cfg);
        j.push_left("k".into(), 100, "L0");
        let _ = j.push_right("k".into(), 300, "R0");
        assert!(j.pane_count() >= 2);
        // wm past everything should purge.
        let purged = j.advance_watermark(10_000);
        assert!(purged >= 2);
        assert_eq!(j.pane_count(), 0);
    }

    #[test]
    fn late_event_dropped_when_all_target_panes_closed() {
        let cfg = TumblingSlidingJoinConfig::new(1_000, 1_000, 500);
        let mut j: TumblingSlidingJoin<&str, &str> = TumblingSlidingJoin::new(cfg);
        j.advance_watermark(10_000);
        let r = j.push_right("k".into(), 500, "Late");
        assert!(r.is_empty());
        assert_eq!(j.stats.late_events_dropped, 1);
    }

    #[test]
    fn allowed_lateness_extends_pane_lifetime() {
        let cfg = TumblingSlidingJoinConfig::new(1_000, 1_000, 500).with_lateness(2_000);
        let mut j: TumblingSlidingJoin<&str, &str> = TumblingSlidingJoin::new(cfg);
        j.push_left("k".into(), 100, "L0");
        j.advance_watermark(2_500); // pane [0,1000) closes at 3000
                                    // Late but within budget → still accepted.
        let r = j.push_right("k".into(), 700, "R0");
        assert_eq!(r.len(), 1);
    }
}
