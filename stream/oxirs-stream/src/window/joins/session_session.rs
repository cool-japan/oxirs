//! Session × Session watermark-driven join.
//!
//! Both sides of the join use *session windows* defined by an inactivity gap.
//! For a given key, consecutive events on either side are coalesced into one
//! session as long as the inter-arrival gap is ≤ `gap_ms`.  When the gap is
//! exceeded a fresh session starts.
//!
//! Two sessions (one per side) join *iff* they share the same key and their
//! intervals (`[min_ts, max_ts]`) overlap.  Output is emitted on session
//! closure: when the watermark advances past `session_end + allowed_lateness`,
//! all matched pairs from that session are emitted and the session is purged.
//!
//! Late events arriving for an already-closed session are dropped and counted.

use std::collections::HashMap;

use super::{WindowJoinKey, WindowJoinResult, WindowJoinStats};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Session × session join configuration.
#[derive(Debug, Clone)]
pub struct SessionSessionJoinConfig {
    /// Inactivity gap that closes a session (ms).  Must be `> 0`.
    pub gap_ms: i64,
    /// Allowed lateness past `session_end` before purge.
    pub allowed_lateness_ms: i64,
}

impl SessionSessionJoinConfig {
    /// Build a config with the supplied gap and zero lateness.
    pub fn new(gap_ms: i64) -> Self {
        assert!(gap_ms > 0, "gap_ms must be > 0");
        Self {
            gap_ms,
            allowed_lateness_ms: 0,
        }
    }

    /// Override allowed lateness.
    pub fn with_lateness(mut self, allowed_lateness_ms: i64) -> Self {
        self.allowed_lateness_ms = allowed_lateness_ms;
        self
    }
}

// ─── Session state ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Session<E: Clone> {
    /// Earliest event timestamp in this session.
    first_ts_ms: i64,
    /// Latest event timestamp in this session (extends with each new event).
    last_ts_ms: i64,
    /// All events in arrival order.
    events: Vec<(i64, E)>,
}

impl<E: Clone> Session<E> {
    fn new(ts_ms: i64, event: E) -> Self {
        Self {
            first_ts_ms: ts_ms,
            last_ts_ms: ts_ms,
            events: vec![(ts_ms, event)],
        }
    }

    fn extend(&mut self, ts_ms: i64, event: E) {
        if ts_ms < self.first_ts_ms {
            self.first_ts_ms = ts_ms;
        }
        if ts_ms > self.last_ts_ms {
            self.last_ts_ms = ts_ms;
        }
        self.events.push((ts_ms, event));
    }

    fn end_ms(&self, gap: i64) -> i64 {
        self.last_ts_ms.saturating_add(gap)
    }

    fn close_at(&self, gap: i64, lateness: i64) -> i64 {
        self.end_ms(gap).saturating_add(lateness)
    }
}

// ─── SessionSessionJoin ──────────────────────────────────────────────────────

/// Watermark-driven session × session join operator.
pub struct SessionSessionJoin<L: Clone, R: Clone> {
    config: SessionSessionJoinConfig,
    left_sessions: HashMap<WindowJoinKey, Vec<Session<L>>>,
    right_sessions: HashMap<WindowJoinKey, Vec<Session<R>>>,
    last_watermark_ms: i64,
    stats: WindowJoinStats,
}

impl<L: Clone, R: Clone> SessionSessionJoin<L, R> {
    /// Create a new join.
    pub fn new(config: SessionSessionJoinConfig) -> Self {
        Self {
            config,
            left_sessions: HashMap::new(),
            right_sessions: HashMap::new(),
            last_watermark_ms: i64::MIN,
            stats: WindowJoinStats::default(),
        }
    }

    fn extend_or_new<E: Clone>(
        sessions: &mut Vec<Session<E>>,
        ts_ms: i64,
        gap: i64,
        event: E,
    ) -> bool {
        // True if we found an existing session within `gap` distance.
        for s in sessions.iter_mut() {
            if ts_ms.saturating_sub(s.last_ts_ms).abs() <= gap
                || ts_ms.saturating_sub(s.first_ts_ms).abs() <= gap
                || (ts_ms >= s.first_ts_ms && ts_ms <= s.last_ts_ms)
            {
                s.extend(ts_ms, event);
                return true;
            }
        }
        sessions.push(Session::new(ts_ms, event));
        false
    }

    fn is_late(&self, ts_ms: i64) -> bool {
        if self.last_watermark_ms == i64::MIN {
            return false;
        }
        // An event is "late" only if the watermark has already advanced past
        // both its potential session end + allowed lateness.
        ts_ms.saturating_add(self.config.gap_ms)
            < self
                .last_watermark_ms
                .saturating_sub(self.config.allowed_lateness_ms)
    }

    /// Insert a left event.
    pub fn push_left(&mut self, key: WindowJoinKey, ts_ms: i64, event: L) {
        if self.is_late(ts_ms) {
            self.stats.late_events_dropped += 1;
            return;
        }
        self.stats.left_events += 1;
        let gap = self.config.gap_ms;
        let entry = self.left_sessions.entry(key).or_default();
        let _ = Self::extend_or_new(entry, ts_ms, gap, event);
    }

    /// Insert a right event.
    pub fn push_right(&mut self, key: WindowJoinKey, ts_ms: i64, event: R) {
        if self.is_late(ts_ms) {
            self.stats.late_events_dropped += 1;
            return;
        }
        self.stats.right_events += 1;
        let gap = self.config.gap_ms;
        let entry = self.right_sessions.entry(key).or_default();
        let _ = Self::extend_or_new(entry, ts_ms, gap, event);
    }

    /// Advance the watermark.  Emits cross-products for pairs of sessions
    /// (one per side, same key) where *both* sides have closed
    /// (`end + allowed_lateness ≤ watermark`).  Closed sessions are then
    /// purged.  If only one side of a key is closed at this watermark, the
    /// closed session is held until its peer also closes — preventing the
    /// case where a fast-closing left side would "lose" its potential right
    /// match.
    pub fn advance_watermark(&mut self, watermark_ms: i64) -> Vec<WindowJoinResult<L, R>> {
        if watermark_ms < self.last_watermark_ms {
            return Vec::new();
        }
        self.last_watermark_ms = watermark_ms;

        let gap = self.config.gap_ms;
        let lat = self.config.allowed_lateness_ms;
        let mut emitted = Vec::new();
        let mut purged = 0usize;

        let keys: Vec<WindowJoinKey> = {
            let mut k: Vec<WindowJoinKey> = self
                .left_sessions
                .keys()
                .chain(self.right_sessions.keys())
                .cloned()
                .collect();
            k.sort();
            k.dedup();
            k
        };

        for key in keys {
            let left_closed_count = self
                .left_sessions
                .get(&key)
                .map(|v| {
                    v.iter()
                        .filter(|s| s.close_at(gap, lat) <= watermark_ms)
                        .count()
                })
                .unwrap_or(0);
            let right_closed_count = self
                .right_sessions
                .get(&key)
                .map(|v| {
                    v.iter()
                        .filter(|s| s.close_at(gap, lat) <= watermark_ms)
                        .count()
                })
                .unwrap_or(0);

            // Only emit + purge when at least one side is fully closed AND
            // every session on the other side that *could* overlap has also
            // closed.  We approximate "could overlap" by requiring the entire
            // other-side bucket to be closed — the standard Flink-style
            // session-join semantics.
            let left_total = self.left_sessions.get(&key).map(|v| v.len()).unwrap_or(0);
            let right_total = self.right_sessions.get(&key).map(|v| v.len()).unwrap_or(0);

            let both_sides_closed = left_closed_count == left_total
                && right_closed_count == right_total
                && left_total > 0
                && right_total > 0;
            if !both_sides_closed {
                continue;
            }

            // Cross product of closed sessions.
            let lefts: Vec<Session<L>> = self.left_sessions.get(&key).cloned().unwrap_or_default();
            let rights: Vec<Session<R>> =
                self.right_sessions.get(&key).cloned().unwrap_or_default();
            for ls in &lefts {
                for rs in &rights {
                    if self.sessions_overlap(ls, rs) {
                        for (_, l_ev) in &ls.events {
                            for (_, r_ev) in &rs.events {
                                emitted.push(WindowJoinResult {
                                    key: key.clone(),
                                    left: l_ev.clone(),
                                    right: r_ev.clone(),
                                    pane_end_ms: ls.end_ms(gap).max(rs.end_ms(gap)),
                                });
                            }
                        }
                    }
                }
            }

            // Purge closed sessions on both sides.
            purged += left_total + right_total;
            self.left_sessions.remove(&key);
            self.right_sessions.remove(&key);
        }

        self.stats.joined_pairs += emitted.len() as u64;
        self.stats.windows_closed += purged as u64;
        emitted
    }

    fn sessions_overlap(&self, a: &Session<L>, b: &Session<R>) -> bool {
        // Two sessions overlap when their gap-extended intervals intersect:
        // [a.first, a.last + gap] ∩ [b.first, b.last + gap] ≠ ∅
        let gap = self.config.gap_ms;
        let a_end = a.last_ts_ms.saturating_add(gap);
        let b_end = b.last_ts_ms.saturating_add(gap);
        a.first_ts_ms <= b_end && b.first_ts_ms <= a_end
    }

    /// Statistics snapshot.
    pub fn stats(&self) -> &WindowJoinStats {
        &self.stats
    }

    /// Number of buffered (unemitted) sessions across both sides.
    pub fn session_count(&self) -> usize {
        self.left_sessions.values().map(|v| v.len()).sum::<usize>()
            + self.right_sessions.values().map(|v| v.len()).sum::<usize>()
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
    fn overlapping_sessions_emit_on_close() {
        let cfg = SessionSessionJoinConfig::new(500);
        let mut j: SessionSessionJoin<&str, &str> = SessionSessionJoin::new(cfg);
        // Left session 100, 200, 300 (last_ts=300, end=800).
        j.push_left("k".into(), 100, "L0");
        j.push_left("k".into(), 200, "L1");
        j.push_left("k".into(), 300, "L2");
        // Right session 250, 350 (last_ts=350, end=850). Overlaps left.
        j.push_right("k".into(), 250, "R0");
        j.push_right("k".into(), 350, "R1");
        // Watermark must pass max(800, 850) = 850 to close both.
        let out = j.advance_watermark(900);
        // 3 left × 2 right = 6 pairs.
        assert_eq!(out.len(), 6);
        assert_eq!(j.session_count(), 0);
    }

    #[test]
    fn non_overlapping_sessions_dont_emit() {
        let cfg = SessionSessionJoinConfig::new(50);
        let mut j: SessionSessionJoin<&str, &str> = SessionSessionJoin::new(cfg);
        j.push_left("k".into(), 100, "L0"); // session L1 ends 150
        j.push_right("k".into(), 1_000, "R0"); // session R1 ends 1050
        let out = j.advance_watermark(2_000);
        assert!(out.is_empty());
    }

    #[test]
    fn separate_keys_dont_join() {
        let cfg = SessionSessionJoinConfig::new(500);
        let mut j: SessionSessionJoin<&str, &str> = SessionSessionJoin::new(cfg);
        j.push_left("a".into(), 100, "La");
        j.push_right("b".into(), 200, "Rb");
        let out = j.advance_watermark(2_000);
        assert!(out.is_empty());
    }

    #[test]
    fn late_event_after_emit_is_dropped() {
        let cfg = SessionSessionJoinConfig::new(50);
        let mut j: SessionSessionJoin<&str, &str> = SessionSessionJoin::new(cfg);
        j.push_left("k".into(), 100, "L0");
        // Advance well past gap so any future event for ts=100 is "late".
        j.advance_watermark(10_000);
        j.push_left("k".into(), 100, "Late");
        assert_eq!(j.stats.late_events_dropped, 1);
    }

    #[test]
    fn allowed_lateness_keeps_session_open() {
        let cfg = SessionSessionJoinConfig::new(50).with_lateness(1_000);
        let mut j: SessionSessionJoin<&str, &str> = SessionSessionJoin::new(cfg);
        j.push_left("k".into(), 100, "L0");
        // Watermark past session end (150) but within lateness budget.
        let out = j.advance_watermark(800);
        assert!(out.is_empty());
        // Late right event still accepted.
        j.push_right("k".into(), 120, "R0");
        // Now advance past lateness budget → emits.
        let out = j.advance_watermark(2_000);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn watermark_emits_only_closed_sessions() {
        let cfg = SessionSessionJoinConfig::new(100);
        let mut j: SessionSessionJoin<&str, &str> = SessionSessionJoin::new(cfg);
        j.push_left("k".into(), 100, "L0"); // session ends 200
        j.push_right("k".into(), 150, "R0"); // session ends 250
                                             // wm = 220 → only left closed, right still open. Emit: nothing yet.
        let out = j.advance_watermark(220);
        assert!(out.is_empty());
        // wm = 260 → both closed.
        let out = j.advance_watermark(260);
        assert_eq!(out.len(), 1);
    }
}
