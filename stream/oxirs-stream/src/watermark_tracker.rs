//! Event-time watermark tracking for stream processing.
//!
//! Watermarks represent the progress of event time in a stream. A watermark W(t) asserts
//! that all events with timestamp <= t have been observed. The global watermark is the
//! minimum across all registered sources.

use std::collections::{HashMap, VecDeque};

/// A watermark update from a single source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Watermark {
    /// Event-time timestamp in milliseconds
    pub timestamp: i64,
    /// Identifier of the source that generated this watermark
    pub source_id: String,
}

impl Watermark {
    /// Create a new watermark.
    pub fn new(source_id: impl Into<String>, timestamp: i64) -> Self {
        Self {
            timestamp,
            source_id: source_id.into(),
        }
    }
}

/// Configuration for watermark tracking behaviour.
#[derive(Debug, Clone)]
pub struct WatermarkConfig {
    /// How many milliseconds late an event may arrive and still be processed
    pub allowed_lateness_ms: i64,
    /// After this many milliseconds without an update a source is considered idle
    pub idle_source_timeout_ms: i64,
    /// Maximum number of historical watermark values to retain
    pub max_history_size: usize,
}

impl Default for WatermarkConfig {
    fn default() -> Self {
        Self {
            allowed_lateness_ms: 5_000,
            idle_source_timeout_ms: 30_000,
            max_history_size: 1_000,
        }
    }
}

/// Watermark state for a single source.
#[derive(Debug, Clone)]
pub struct SourceWatermark {
    /// Identifier of the source
    pub source_id: String,
    /// Current watermark timestamp for this source
    pub current: i64,
    /// Wall-clock time (ms) when this source was last updated; `None` means never updated
    pub last_updated_opt: Option<i64>,
}

impl SourceWatermark {
    fn new(source_id: impl Into<String>) -> Self {
        Self {
            source_id: source_id.into(),
            current: i64::MIN,
            last_updated_opt: None,
        }
    }

    /// Wall-clock time of last update, or 0 if never updated (for public API compatibility).
    pub fn last_updated(&self) -> i64 {
        self.last_updated_opt.unwrap_or(0)
    }
}

/// Tracks event-time watermarks across multiple sources.
///
/// The global watermark is the minimum watermark across all registered sources.
/// When a source advances its watermark the global watermark may advance too.
#[derive(Debug)]
pub struct WatermarkTracker {
    sources: HashMap<String, SourceWatermark>,
    global_watermark: i64,
    config: WatermarkConfig,
    watermark_history: VecDeque<i64>,
}

impl WatermarkTracker {
    /// Create a new tracker with the given configuration.
    pub fn new(config: WatermarkConfig) -> Self {
        Self {
            sources: HashMap::new(),
            global_watermark: i64::MIN,
            config,
            watermark_history: VecDeque::new(),
        }
    }

    /// Register a new source.  The source starts with watermark = i64::MIN.
    pub fn register_source(&mut self, source_id: impl Into<String>) {
        let id: String = source_id.into();
        self.sources
            .entry(id.clone())
            .or_insert_with(|| SourceWatermark::new(id));
        // Re-compute global watermark after adding a source (it may decrease)
        self.recompute_global();
    }

    /// Update the watermark for a source.
    ///
    /// Returns `Some(new_global)` if the global watermark advanced, otherwise `None`.
    pub fn update(&mut self, watermark: Watermark, current_time_ms: i64) -> Option<i64> {
        let old_global = self.global_watermark;

        let source = self
            .sources
            .entry(watermark.source_id.clone())
            .or_insert_with(|| SourceWatermark::new(watermark.source_id.clone()));

        // Watermarks must be monotonically increasing per source
        if watermark.timestamp > source.current {
            source.current = watermark.timestamp;
        }
        source.last_updated_opt = Some(current_time_ms);

        self.recompute_global();

        if self.global_watermark > old_global {
            self.record_history(self.global_watermark);
            Some(self.global_watermark)
        } else {
            None
        }
    }

    /// Return the current global watermark.
    pub fn global_watermark(&self) -> i64 {
        self.global_watermark
    }

    /// Return the current watermark for a specific source.
    pub fn source_watermark(&self, source_id: &str) -> Option<i64> {
        self.sources.get(source_id).map(|s| s.current)
    }

    /// Deregister sources that have not emitted a watermark update within the idle timeout.
    ///
    /// Returns the IDs of the removed sources.
    pub fn deregister_idle_sources(&mut self, current_time_ms: i64) -> Vec<String> {
        let timeout = self.config.idle_source_timeout_ms;
        let idle: Vec<String> = self
            .sources
            .iter()
            .filter(|(_, s)| {
                // A source is idle when it has received at least one update but has not
                // sent another within the timeout window.
                if let Some(last) = s.last_updated_opt {
                    (current_time_ms - last) >= timeout
                } else {
                    false // never updated → not idle
                }
            })
            .map(|(id, _)| id.clone())
            .collect();

        for id in &idle {
            self.sources.remove(id);
        }

        if !idle.is_empty() {
            let old_global = self.global_watermark;
            self.recompute_global();
            if self.global_watermark > old_global {
                self.record_history(self.global_watermark);
            }
        }

        idle
    }

    /// Return the total number of registered sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Return the number of sources that emitted an update within the idle timeout.
    pub fn active_source_count(&self, current_time_ms: i64) -> usize {
        let timeout = self.config.idle_source_timeout_ms;
        self.sources
            .values()
            .filter(|s| {
                // If the source has never emitted an update it is considered active
                // (it may just be warming up).
                match s.last_updated_opt {
                    None => true, // never updated → active
                    Some(last) => (current_time_ms - last) < timeout,
                }
            })
            .count()
    }

    /// Return the recorded global watermark history.
    pub fn watermark_history(&self) -> &VecDeque<i64> {
        &self.watermark_history
    }

    /// Return `true` if the event arrived too late to be processed.
    ///
    /// An event is late when `event_time < global_watermark - allowed_lateness_ms`.
    pub fn is_late(&self, event_time: i64) -> bool {
        if self.global_watermark == i64::MIN {
            return false;
        }
        event_time < self.global_watermark - self.config.allowed_lateness_ms
    }

    /// Force-advance the global watermark to `timestamp`.
    ///
    /// Returns `true` if the watermark actually advanced.
    pub fn advance_to(&mut self, timestamp: i64) -> bool {
        if timestamp > self.global_watermark {
            self.global_watermark = timestamp;
            self.record_history(timestamp);
            true
        } else {
            false
        }
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn recompute_global(&mut self) {
        if self.sources.is_empty() {
            // No sources — keep the existing global watermark (don't lower it)
            return;
        }
        let min = self
            .sources
            .values()
            .map(|s| s.current)
            .min()
            .unwrap_or(i64::MIN);
        // The global watermark must not decrease below its current level,
        // but it CAN advance when the minimum of remaining sources moves up.
        if min > self.global_watermark {
            self.global_watermark = min;
        }
    }

    fn record_history(&mut self, value: i64) {
        if self.watermark_history.len() >= self.config.max_history_size {
            self.watermark_history.pop_front();
        }
        self.watermark_history.push_back(value);
    }
}

impl Default for WatermarkTracker {
    fn default() -> Self {
        Self::new(WatermarkConfig::default())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> WatermarkConfig {
        WatermarkConfig {
            allowed_lateness_ms: 100,
            idle_source_timeout_ms: 1_000,
            max_history_size: 50,
        }
    }

    fn wm(source: &str, ts: i64) -> Watermark {
        Watermark::new(source, ts)
    }

    // 1. Single source watermark advances
    #[test]
    fn test_single_source_watermark_advance() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.register_source("s1");

        let result = tracker.update(wm("s1", 100), 0);
        assert_eq!(result, Some(100));
        assert_eq!(tracker.global_watermark(), 100);
    }

    // 2. Watermarks must be monotonically increasing per source
    #[test]
    fn test_single_source_monotonic() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.register_source("s1");

        tracker.update(wm("s1", 200), 0);
        let result = tracker.update(wm("s1", 100), 1); // older ts — should not advance
        assert_eq!(result, None);
        assert_eq!(tracker.global_watermark(), 200);
    }

    // 3. Multi-source: global is minimum
    #[test]
    fn test_multi_source_minimum() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.register_source("s1");
        tracker.register_source("s2");

        tracker.update(wm("s1", 500), 0);
        assert_eq!(tracker.global_watermark(), i64::MIN); // s2 still at MIN

        tracker.update(wm("s2", 200), 0);
        assert_eq!(tracker.global_watermark(), 200); // min(500, 200) = 200
    }

    // 4. Multi-source: both advance so global advances
    #[test]
    fn test_multi_source_both_advance() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.register_source("s1");
        tracker.register_source("s2");

        tracker.update(wm("s1", 100), 0);
        tracker.update(wm("s2", 100), 0);
        assert_eq!(tracker.global_watermark(), 100);

        tracker.update(wm("s1", 300), 1);
        tracker.update(wm("s2", 300), 1);
        assert_eq!(tracker.global_watermark(), 300);
    }

    // 5. Idle source removal
    #[test]
    fn test_idle_source_removal() {
        let mut tracker = WatermarkTracker::new(WatermarkConfig {
            idle_source_timeout_ms: 1_000,
            ..cfg()
        });
        tracker.register_source("s1");
        tracker.register_source("s2");

        tracker.update(wm("s1", 100), 0);
        tracker.update(wm("s2", 200), 0);

        // Advance time by 2 seconds — both sources are now idle
        let removed = tracker.deregister_idle_sources(2_000);
        assert_eq!(removed.len(), 2);
        assert_eq!(tracker.source_count(), 0);
    }

    // 6. deregister_idle_sources returns correct ids
    #[test]
    fn test_deregister_returns_correct_ids() {
        let mut tracker = WatermarkTracker::new(WatermarkConfig {
            idle_source_timeout_ms: 1_000,
            ..cfg()
        });
        tracker.register_source("s1");
        tracker.register_source("s2");

        // Only update s1 at t=0
        tracker.update(wm("s1", 100), 0);
        tracker.update(wm("s2", 200), 0); // s2 updated at t=0

        // At t=1500 both are idle
        let mut removed = tracker.deregister_idle_sources(1_500);
        removed.sort();
        assert!(removed.contains(&"s1".to_string()));
        assert!(removed.contains(&"s2".to_string()));
    }

    // 7. Sources that were never updated are NOT considered idle
    #[test]
    fn test_never_updated_not_idle() {
        let mut tracker = WatermarkTracker::new(WatermarkConfig {
            idle_source_timeout_ms: 1_000,
            ..cfg()
        });
        tracker.register_source("s1");

        // s1 was registered but never had update() called
        let removed = tracker.deregister_idle_sources(5_000);
        assert!(removed.is_empty());
    }

    // 8. Late event detection
    #[test]
    fn test_late_event_detection() {
        let mut tracker = WatermarkTracker::new(WatermarkConfig {
            allowed_lateness_ms: 100,
            ..cfg()
        });
        tracker.register_source("s1");
        tracker.update(wm("s1", 1_000), 0);

        // event at 895 is late (1000 - 100 = 900, 895 < 900)
        assert!(tracker.is_late(895));
        // event at 905 is NOT late
        assert!(!tracker.is_late(905));
        // event at exactly the boundary is not late
        assert!(!tracker.is_late(900));
    }

    // 9. Not late when global watermark is MIN
    #[test]
    fn test_not_late_when_no_watermark() {
        let tracker = WatermarkTracker::new(cfg());
        assert!(!tracker.is_late(0));
        assert!(!tracker.is_late(i64::MIN));
    }

    // 10. advance_to advances
    #[test]
    fn test_advance_to() {
        let mut tracker = WatermarkTracker::new(cfg());
        let advanced = tracker.advance_to(500);
        assert!(advanced);
        assert_eq!(tracker.global_watermark(), 500);
    }

    // 11. advance_to does not go backwards
    #[test]
    fn test_advance_to_no_backward() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.advance_to(500);
        let advanced = tracker.advance_to(100);
        assert!(!advanced);
        assert_eq!(tracker.global_watermark(), 500);
    }

    // 12. advance_to to same value returns false
    #[test]
    fn test_advance_to_same_value() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.advance_to(500);
        let result = tracker.advance_to(500);
        assert!(!result);
    }

    // 13. History recording
    #[test]
    fn test_history_recording() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.register_source("s1");

        tracker.update(wm("s1", 100), 0);
        tracker.update(wm("s1", 200), 1);
        tracker.update(wm("s1", 300), 2);

        let h = tracker.watermark_history();
        assert_eq!(h.len(), 3);
        assert_eq!(*h.front().unwrap(), 100);
        assert_eq!(*h.back().unwrap(), 300);
    }

    // 14. History capped at max_history_size
    #[test]
    fn test_history_capped() {
        let mut tracker = WatermarkTracker::new(WatermarkConfig {
            max_history_size: 5,
            ..cfg()
        });
        tracker.register_source("s1");

        for i in 1..=10 {
            tracker.update(wm("s1", i * 100), i as i64);
        }

        assert_eq!(tracker.watermark_history().len(), 5);
    }

    // 15. advance_to adds to history
    #[test]
    fn test_advance_to_history() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.advance_to(1_000);
        assert_eq!(tracker.watermark_history().len(), 1);
        assert_eq!(*tracker.watermark_history().front().unwrap(), 1_000);
    }

    // 16. source_watermark returns correct value
    #[test]
    fn test_source_watermark() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.register_source("s1");
        assert_eq!(tracker.source_watermark("s1"), Some(i64::MIN));

        tracker.update(wm("s1", 300), 0);
        assert_eq!(tracker.source_watermark("s1"), Some(300));
    }

    // 17. source_watermark for unknown source returns None
    #[test]
    fn test_source_watermark_unknown() {
        let tracker = WatermarkTracker::new(cfg());
        assert_eq!(tracker.source_watermark("unknown"), None);
    }

    // 18. source_count
    #[test]
    fn test_source_count() {
        let mut tracker = WatermarkTracker::new(cfg());
        assert_eq!(tracker.source_count(), 0);
        tracker.register_source("s1");
        tracker.register_source("s2");
        assert_eq!(tracker.source_count(), 2);
    }

    // 19. active_source_count
    #[test]
    fn test_active_source_count() {
        let mut tracker = WatermarkTracker::new(WatermarkConfig {
            idle_source_timeout_ms: 1_000,
            ..cfg()
        });
        tracker.register_source("s1");
        tracker.register_source("s2");

        // Neither has been updated — both considered active (last_updated_opt == None)
        assert_eq!(tracker.active_source_count(0), 2);

        // Update s1 at t=0, then check at t=2000 — s1 is now idle
        tracker.update(wm("s1", 100), 0);
        assert_eq!(tracker.active_source_count(2_000), 1); // only s2 (never updated)
    }

    // 20. Global watermark minimum of all sources
    #[test]
    fn test_global_is_minimum_of_sources() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.register_source("s1");
        tracker.register_source("s2");
        tracker.register_source("s3");

        tracker.update(wm("s1", 1_000), 0);
        tracker.update(wm("s2", 500), 0);
        tracker.update(wm("s3", 750), 0);

        assert_eq!(tracker.global_watermark(), 500);
    }

    // 21. After source removal, global may advance
    #[test]
    fn test_global_advances_after_source_removal() {
        let mut tracker = WatermarkTracker::new(WatermarkConfig {
            idle_source_timeout_ms: 1_000,
            ..cfg()
        });
        tracker.register_source("s1");
        tracker.register_source("s2");

        tracker.update(wm("s1", 1_000), 0);
        tracker.update(wm("s2", 50), 0); // slow source

        assert_eq!(tracker.global_watermark(), 50);

        // Keep s1 active by updating it at t=1500 (within timeout)
        // Only s2 stays idle (last updated at t=0, now t=2000 > 1000ms timeout)
        tracker.update(wm("s1", 1_500), 1_500);

        // s2 goes idle (last updated t=0, check at t=2000)
        let removed = tracker.deregister_idle_sources(2_000);
        assert!(removed.contains(&"s2".to_string()));

        // Now only s1 remains — global should reflect s1's watermark (1500)
        assert_eq!(tracker.global_watermark(), 1_500);
    }

    // 22. register same source twice is idempotent
    #[test]
    fn test_register_source_idempotent() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.register_source("s1");
        tracker.register_source("s1");
        assert_eq!(tracker.source_count(), 1);
    }

    // 23. update auto-registers unknown source
    #[test]
    fn test_update_auto_registers_source() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.update(wm("new_source", 100), 0);
        assert!(tracker.source_watermark("new_source").is_some());
    }

    // 24. is_late with exactly allowed lateness boundary
    #[test]
    fn test_is_late_boundary() {
        let mut tracker = WatermarkTracker::new(WatermarkConfig {
            allowed_lateness_ms: 200,
            ..cfg()
        });
        tracker.advance_to(1_000);

        // Boundary: 1000 - 200 = 800
        assert!(!tracker.is_late(800)); // exactly at boundary — not late
        assert!(tracker.is_late(799));  // one ms before boundary — late
    }

    // 25. advance_to then source update
    #[test]
    fn test_advance_to_then_source_update() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.advance_to(1_000);
        tracker.register_source("s1");
        // s1 registers with MIN; global should not drop below advance_to value
        assert_eq!(tracker.global_watermark(), 1_000);
    }

    // 26. update returns None when source does not advance global
    #[test]
    fn test_update_returns_none_no_global_advance() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.register_source("s1");
        tracker.register_source("s2");

        tracker.update(wm("s2", 1_000), 0);
        // s1 still at MIN so global stays at MIN — no advancement
        let result = tracker.update(wm("s2", 2_000), 1);
        assert_eq!(result, None);
    }

    // 27. Watermark struct equality
    #[test]
    fn test_watermark_equality() {
        let w1 = Watermark::new("s1", 100);
        let w2 = Watermark::new("s1", 100);
        assert_eq!(w1, w2);
    }

    // 28. Default tracker
    #[test]
    fn test_default_tracker() {
        let tracker = WatermarkTracker::default();
        assert_eq!(tracker.global_watermark(), i64::MIN);
        assert_eq!(tracker.source_count(), 0);
    }

    // 29. Empty history initially
    #[test]
    fn test_empty_history_initially() {
        let tracker = WatermarkTracker::new(cfg());
        assert!(tracker.watermark_history().is_empty());
    }

    // 30. Multiple sources, one advances but global stays at slow source
    #[test]
    fn test_fast_source_does_not_move_global_past_slow() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.register_source("fast");
        tracker.register_source("slow");

        tracker.update(wm("slow", 10), 0);
        tracker.update(wm("fast", 10), 0);

        for i in 1..=10 {
            tracker.update(wm("fast", i * 1_000), i as i64);
        }

        assert_eq!(tracker.global_watermark(), 10);
    }

    // 31. deregister_idle_sources with no idle sources
    #[test]
    fn test_deregister_no_idle() {
        let mut tracker = WatermarkTracker::new(WatermarkConfig {
            idle_source_timeout_ms: 10_000,
            ..cfg()
        });
        tracker.register_source("s1");
        tracker.update(wm("s1", 100), 0);
        let removed = tracker.deregister_idle_sources(5_000);
        assert!(removed.is_empty());
    }

    // 32. active_source_count with zero sources
    #[test]
    fn test_active_source_count_empty() {
        let tracker = WatermarkTracker::new(cfg());
        assert_eq!(tracker.active_source_count(0), 0);
    }

    // 33. source_count after deregister
    #[test]
    fn test_source_count_after_deregister() {
        let mut tracker = WatermarkTracker::new(WatermarkConfig {
            idle_source_timeout_ms: 1_000,
            ..cfg()
        });
        tracker.register_source("s1");
        tracker.update(wm("s1", 100), 0);
        tracker.deregister_idle_sources(5_000);
        assert_eq!(tracker.source_count(), 0);
    }

    // 34. history does not grow when watermark does not advance
    #[test]
    fn test_history_does_not_grow_on_no_advance() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.register_source("s1");
        tracker.register_source("s2"); // s2 blocks global

        tracker.update(wm("s1", 500), 0);
        // Global not advanced (s2 still at MIN)
        assert!(tracker.watermark_history().is_empty());
    }

    // 35. is_late always false with zero allowed lateness is_late == event < global
    #[test]
    fn test_is_late_zero_lateness() {
        let mut tracker = WatermarkTracker::new(WatermarkConfig {
            allowed_lateness_ms: 0,
            ..cfg()
        });
        tracker.advance_to(1_000);
        assert!(tracker.is_late(999));
        assert!(!tracker.is_late(1_000));
    }

    // 36. WatermarkConfig default values
    #[test]
    fn test_watermark_config_default() {
        let cfg = WatermarkConfig::default();
        assert_eq!(cfg.allowed_lateness_ms, 5_000);
        assert_eq!(cfg.idle_source_timeout_ms, 30_000);
        assert_eq!(cfg.max_history_size, 1_000);
    }

    // 37. SourceWatermark starts at MIN
    #[test]
    fn test_source_watermark_starts_at_min() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.register_source("s1");
        assert_eq!(tracker.source_watermark("s1"), Some(i64::MIN));
    }

    // 38. Three sources, remove slowest, global advances to second slowest
    #[test]
    fn test_remove_slowest_source_global_advances() {
        let mut tracker = WatermarkTracker::new(WatermarkConfig {
            idle_source_timeout_ms: 1_000,
            ..cfg()
        });
        tracker.register_source("s1");
        tracker.register_source("s2");
        tracker.register_source("s3");

        tracker.update(wm("s1", 100), 0);  // slowest
        tracker.update(wm("s2", 500), 0);
        tracker.update(wm("s3", 900), 0);

        assert_eq!(tracker.global_watermark(), 100);

        // Only s1 becomes idle
        tracker.update(wm("s2", 600), 2_000);
        tracker.update(wm("s3", 1_000), 2_000);
        let removed = tracker.deregister_idle_sources(2_000);
        assert!(removed.contains(&"s1".to_string()));

        assert_eq!(tracker.global_watermark(), 600);
    }

    // 39. advance_to advances history len
    #[test]
    fn test_advance_to_increments_history() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.advance_to(1_000);
        tracker.advance_to(2_000);
        tracker.advance_to(3_000);
        assert_eq!(tracker.watermark_history().len(), 3);
    }

    // 40. global watermark advanced by source update is recorded in history
    #[test]
    fn test_history_recorded_on_source_advance() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.register_source("s1");
        tracker.update(wm("s1", 1_000), 0);
        assert_eq!(tracker.watermark_history().len(), 1);
        assert_eq!(*tracker.watermark_history().back().unwrap(), 1_000);
    }

    // 41. Multiple updates same timestamp only records one history entry
    #[test]
    fn test_history_no_duplicate_same_ts() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.register_source("s1");
        tracker.update(wm("s1", 500), 0);
        // Same timestamp again — source doesn't advance so no history entry
        tracker.update(wm("s1", 500), 1);
        assert_eq!(tracker.watermark_history().len(), 1);
    }

    // 42. is_late returns false when global is MIN
    #[test]
    fn test_is_late_global_min() {
        let tracker = WatermarkTracker::new(cfg());
        assert!(!tracker.is_late(i64::MAX));
    }

    // 43. source_watermark reflects most recent update
    #[test]
    fn test_source_watermark_reflects_latest() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.register_source("s1");
        tracker.update(wm("s1", 100), 0);
        tracker.update(wm("s1", 200), 1);
        tracker.update(wm("s1", 300), 2);
        assert_eq!(tracker.source_watermark("s1"), Some(300));
    }

    // 44. global watermark after advance_to and source registration
    #[test]
    fn test_source_registered_after_advance_to() {
        let mut tracker = WatermarkTracker::new(cfg());
        tracker.advance_to(500);
        tracker.register_source("s1");
        // Adding s1 with MIN may or may not reduce global; our implementation
        // ensures the global never decreases from advance_to
        assert_eq!(tracker.global_watermark(), 500);
    }

    // 45. active_source_count excludes deregistered sources
    #[test]
    fn test_active_count_after_deregister() {
        let mut tracker = WatermarkTracker::new(WatermarkConfig {
            idle_source_timeout_ms: 1_000,
            ..cfg()
        });
        tracker.register_source("s1");
        tracker.register_source("s2");

        tracker.update(wm("s1", 100), 0);
        tracker.update(wm("s2", 200), 0);

        tracker.deregister_idle_sources(5_000);
        assert_eq!(tracker.active_source_count(5_000), 0);
    }

    // 46. Watermark new helper
    #[test]
    fn test_watermark_new_helper() {
        let w = Watermark::new("src", 42);
        assert_eq!(w.source_id, "src");
        assert_eq!(w.timestamp, 42);
    }

    // 47. Large number of sources, global is always minimum
    #[test]
    fn test_many_sources_global_is_minimum() {
        let mut tracker = WatermarkTracker::new(cfg());
        for i in 1..=20 {
            tracker.register_source(format!("s{i}"));
            tracker.update(wm(&format!("s{i}"), i * 100), 0);
        }
        // Minimum is s1 = 100
        assert_eq!(tracker.global_watermark(), 100);
    }
}
