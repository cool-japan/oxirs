//! Tumbling-Tumbling watermark-driven join.
//!
//! Both inputs are bucketed into the same fixed-size tumbling pane.  Two
//! events join *iff* they share both the join key and the pane index.
//!
//! Pane closure is *watermark driven*: a pane `[start, start + size)` closes
//! and is purged once `watermark_ms ≥ start + size + allowed_lateness_ms`.
//!
//! Late events arriving for an already-closed pane are dropped and counted
//! in [`super::WindowJoinStats::late_events_dropped`].

use std::collections::HashMap;

use super::{WindowJoinKey, WindowJoinResult, WindowJoinStats};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Tumbling-tumbling join configuration.
#[derive(Debug, Clone)]
pub struct TumblingTumblingJoinConfig {
    /// Tumbling window size in milliseconds (must be `> 0`).
    pub size_ms: i64,
    /// Allowed lateness past `pane_end` before a pane is purged.
    pub allowed_lateness_ms: i64,
}

impl TumblingTumblingJoinConfig {
    /// Build a config with the given size and zero lateness.
    pub fn new(size_ms: i64) -> Self {
        Self {
            size_ms,
            allowed_lateness_ms: 0,
        }
    }

    /// Override allowed lateness.
    pub fn with_lateness(mut self, allowed_lateness_ms: i64) -> Self {
        self.allowed_lateness_ms = allowed_lateness_ms;
        self
    }
}

// ─── Pane state ──────────────────────────────────────────────────────────────

struct PaneState<L, R> {
    pane_start_ms: i64,
    left: HashMap<WindowJoinKey, Vec<L>>,
    right: HashMap<WindowJoinKey, Vec<R>>,
}

impl<L, R> PaneState<L, R> {
    fn new(pane_start_ms: i64) -> Self {
        Self {
            pane_start_ms,
            left: HashMap::new(),
            right: HashMap::new(),
        }
    }
}

// ─── TumblingTumblingJoin ────────────────────────────────────────────────────

/// Watermark-driven tumbling-tumbling join operator.
///
/// `L`, `R` — left/right event payloads (must implement `Clone` so emissions
/// can be paired without consuming buffer state).
pub struct TumblingTumblingJoin<L: Clone, R: Clone> {
    config: TumblingTumblingJoinConfig,
    panes: HashMap<i64, PaneState<L, R>>, // pane_start_ms → state
    last_watermark_ms: i64,
    stats: WindowJoinStats,
}

impl<L: Clone, R: Clone> TumblingTumblingJoin<L, R> {
    /// Create a new join.
    pub fn new(config: TumblingTumblingJoinConfig) -> Self {
        Self {
            config,
            panes: HashMap::new(),
            last_watermark_ms: i64::MIN,
            stats: WindowJoinStats::default(),
        }
    }

    fn pane_start(&self, ts_ms: i64) -> i64 {
        // Floor division that handles negative timestamps correctly.
        let s = self.config.size_ms;
        let q = ts_ms.div_euclid(s);
        q * s
    }

    /// Insert a left event.  Emits any joins that match within the same pane.
    pub fn push_left(
        &mut self,
        key: WindowJoinKey,
        ts_ms: i64,
        event: L,
    ) -> Vec<WindowJoinResult<L, R>> {
        let pane_start = self.pane_start(ts_ms);
        if self.is_closed(pane_start) {
            self.stats.late_events_dropped += 1;
            return Vec::new();
        }
        self.stats.left_events += 1;

        let pane = self
            .panes
            .entry(pane_start)
            .or_insert_with(|| PaneState::new(pane_start));
        let pane_end = pane_start + self.config.size_ms;

        // Match against every right event already in the pane with the same key.
        let mut emitted = Vec::new();
        if let Some(rights) = pane.right.get(&key) {
            for r in rights {
                emitted.push(WindowJoinResult {
                    key: key.clone(),
                    left: event.clone(),
                    right: r.clone(),
                    pane_end_ms: pane_end,
                });
            }
        }
        pane.left.entry(key).or_default().push(event);
        self.stats.joined_pairs += emitted.len() as u64;
        emitted
    }

    /// Insert a right event.  Emits any joins that match within the same pane.
    pub fn push_right(
        &mut self,
        key: WindowJoinKey,
        ts_ms: i64,
        event: R,
    ) -> Vec<WindowJoinResult<L, R>> {
        let pane_start = self.pane_start(ts_ms);
        if self.is_closed(pane_start) {
            self.stats.late_events_dropped += 1;
            return Vec::new();
        }
        self.stats.right_events += 1;

        let pane = self
            .panes
            .entry(pane_start)
            .or_insert_with(|| PaneState::new(pane_start));
        let pane_end = pane_start + self.config.size_ms;

        let mut emitted = Vec::new();
        if let Some(lefts) = pane.left.get(&key) {
            for l in lefts {
                emitted.push(WindowJoinResult {
                    key: key.clone(),
                    left: l.clone(),
                    right: event.clone(),
                    pane_end_ms: pane_end,
                });
            }
        }
        pane.right.entry(key).or_default().push(event);
        self.stats.joined_pairs += emitted.len() as u64;
        emitted
    }

    /// Returns `true` iff the pane that starts at `pane_start_ms` is closed
    /// according to the latest observed watermark.
    fn is_closed(&self, pane_start_ms: i64) -> bool {
        let pane_end = pane_start_ms.saturating_add(self.config.size_ms);
        let close_at = pane_end.saturating_add(self.config.allowed_lateness_ms);
        self.last_watermark_ms >= close_at
    }

    /// Advance the input watermark.  Closes panes whose budget is exhausted.
    /// Returns the number of panes purged.
    pub fn advance_watermark(&mut self, watermark_ms: i64) -> usize {
        if watermark_ms < self.last_watermark_ms {
            // Non-monotonic input — caller violated contract; ignore.
            return 0;
        }
        self.last_watermark_ms = watermark_ms;
        let lateness = self.config.allowed_lateness_ms;
        let size = self.config.size_ms;
        let mut purged = 0usize;
        self.panes.retain(|&pane_start, _| {
            let pane_end = pane_start.saturating_add(size);
            let close_at = pane_end.saturating_add(lateness);
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

    /// Number of currently buffered panes.
    pub fn pane_count(&self) -> usize {
        self.panes.len()
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
    fn joins_events_in_same_pane() {
        let cfg = TumblingTumblingJoinConfig::new(1_000);
        let mut j: TumblingTumblingJoin<&str, &str> = TumblingTumblingJoin::new(cfg);
        // Left at 100ms (pane [0,1000)).
        let r0 = j.push_left("k1".into(), 100, "L0");
        assert!(r0.is_empty());
        // Right at 800ms same pane → join emits.
        let r1 = j.push_right("k1".into(), 800, "R0");
        assert_eq!(r1.len(), 1);
        assert_eq!(r1[0].pane_end_ms, 1_000);
    }

    #[test]
    fn does_not_join_across_panes() {
        let cfg = TumblingTumblingJoinConfig::new(1_000);
        let mut j: TumblingTumblingJoin<&str, &str> = TumblingTumblingJoin::new(cfg);
        j.push_left("k".into(), 100, "L"); // pane 0
        let r = j.push_right("k".into(), 1_500, "R"); // pane 1
        assert!(r.is_empty());
        assert_eq!(j.stats.joined_pairs, 0);
    }

    #[test]
    fn watermark_closes_panes() {
        let cfg = TumblingTumblingJoinConfig::new(1_000);
        let mut j: TumblingTumblingJoin<&str, &str> = TumblingTumblingJoin::new(cfg);
        j.push_left("k".into(), 100, "L0");
        j.push_left("k".into(), 1_500, "L1");
        assert_eq!(j.pane_count(), 2);
        // Advance watermark past pane 0 → close it.
        let purged = j.advance_watermark(1_001);
        assert_eq!(purged, 1);
        assert_eq!(j.pane_count(), 1);
        assert_eq!(j.stats.windows_closed, 1);
    }

    #[test]
    fn late_events_dropped() {
        let cfg = TumblingTumblingJoinConfig::new(1_000);
        let mut j: TumblingTumblingJoin<&str, &str> = TumblingTumblingJoin::new(cfg);
        j.advance_watermark(2_000); // closes panes [0,1000) and [1000,2000)
        let out = j.push_left("k".into(), 500, "Late");
        assert!(out.is_empty());
        assert_eq!(j.stats.late_events_dropped, 1);
    }

    #[test]
    fn allowed_lateness_keeps_window_open() {
        let cfg = TumblingTumblingJoinConfig::new(1_000).with_lateness(500);
        let mut j: TumblingTumblingJoin<&str, &str> = TumblingTumblingJoin::new(cfg);
        j.push_left("k".into(), 100, "L0");
        j.advance_watermark(1_499); // pane closes at 1500
                                    // Late event at 600ms is still accepted.
        let out = j.push_right("k".into(), 600, "R0");
        assert_eq!(out.len(), 1);
        // After 1500 the pane is closed.
        j.advance_watermark(1_501);
        let out = j.push_right("k".into(), 700, "R1");
        assert!(out.is_empty());
        assert_eq!(j.stats.late_events_dropped, 1);
    }

    #[test]
    fn multiple_keys_isolated() {
        let cfg = TumblingTumblingJoinConfig::new(1_000);
        let mut j: TumblingTumblingJoin<&str, &str> = TumblingTumblingJoin::new(cfg);
        j.push_left("a".into(), 100, "La");
        j.push_left("b".into(), 200, "Lb");
        let r1 = j.push_right("a".into(), 300, "Ra");
        assert_eq!(r1.len(), 1);
        let r2 = j.push_right("b".into(), 400, "Rb");
        assert_eq!(r2.len(), 1);
        let r3 = j.push_right("c".into(), 500, "Rc");
        assert!(r3.is_empty());
    }
}
