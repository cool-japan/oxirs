//! Late event handler with drop / re-assign / side-output policies.
//!
//! This file is the dedicated home for the [`LateDataHandler`] type and its
//! related enums.  The original implementation lives in [`super::mod`] and is
//! re-exported through this module so callers can import a stable path:
//!
//! ```rust
//! use oxirs_stream::watermark::late_handler::{LateDataHandler, LateDataPolicy};
//! ```
//!
//! No duplication: [`super::LateDataHandler`], [`super::LateDataPolicy`], and
//! [`super::LateDataDecision`] are simply re-exported.  Allowed-lateness
//! policies, drop counters, and side-output routing all live in the parent
//! module.
//!
//! Additional helpers in this module:
//!
//! * [`AllowedLatenessTracker`] — maintains a per-window allowed-lateness
//!   budget and answers "should we keep the window open for this late event?".
//! * [`SideOutputRouter`] — accumulates events tagged for side-output channels
//!   so callers can drain them per channel.

use std::collections::HashMap;

pub use super::{LateDataDecision, LateDataHandler, LateDataPolicy};

// ─── AllowedLatenessTracker ──────────────────────────────────────────────────

/// Tracks per-window allowed-lateness budgets.
///
/// A window is *closed* once `now ≥ window_end + allowed_lateness`.  Until
/// then late events may still be re-assigned to it.
#[derive(Debug, Default)]
pub struct AllowedLatenessTracker {
    /// Map from window-id → (window_end_ms, allowed_lateness_ms).
    windows: HashMap<String, (i64, i64)>,
}

impl AllowedLatenessTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self {
            windows: HashMap::new(),
        }
    }

    /// Register a window.
    pub fn register(
        &mut self,
        window_id: impl Into<String>,
        window_end_ms: i64,
        allowed_lateness_ms: i64,
    ) {
        self.windows
            .insert(window_id.into(), (window_end_ms, allowed_lateness_ms));
    }

    /// Returns `true` if the window is still accepting late events at `now_ms`.
    pub fn is_open(&self, window_id: &str, now_ms: i64) -> bool {
        match self.windows.get(window_id) {
            None => false,
            Some(&(end_ms, lateness_ms)) => now_ms < end_ms.saturating_add(lateness_ms),
        }
    }

    /// Drop windows whose allowed-lateness budget has expired.
    /// Returns the IDs of evicted windows.
    pub fn evict_closed(&mut self, now_ms: i64) -> Vec<String> {
        let mut evicted = Vec::new();
        self.windows.retain(|id, &mut (end_ms, lateness_ms)| {
            let still_open = now_ms < end_ms.saturating_add(lateness_ms);
            if !still_open {
                evicted.push(id.clone());
            }
            still_open
        });
        evicted
    }

    /// Number of tracked windows.
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    /// True iff no windows are tracked.
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }
}

// ─── SideOutputRouter ────────────────────────────────────────────────────────

/// Per-channel buffer for events routed via [`LateDataPolicy::SideOutput`].
#[derive(Debug, Default)]
pub struct SideOutputRouter<E> {
    channels: HashMap<String, Vec<E>>,
}

impl<E> SideOutputRouter<E> {
    /// Create an empty router.
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
        }
    }

    /// Append `event` to the named channel buffer.
    pub fn push(&mut self, channel: &str, event: E) {
        self.channels
            .entry(channel.to_string())
            .or_default()
            .push(event);
    }

    /// Drain (and return) all events from the given channel.
    pub fn drain(&mut self, channel: &str) -> Vec<E> {
        self.channels.remove(channel).unwrap_or_default()
    }

    /// Number of events buffered on `channel`.
    pub fn len(&self, channel: &str) -> usize {
        self.channels.get(channel).map(|v| v.len()).unwrap_or(0)
    }

    /// True iff the named channel has no buffered events.
    pub fn is_empty(&self, channel: &str) -> bool {
        self.channels
            .get(channel)
            .map(|v| v.is_empty())
            .unwrap_or(true)
    }

    /// All channel names known to this router.
    pub fn channels(&self) -> impl Iterator<Item = &String> {
        self.channels.keys()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_lateness_open_until_budget_exhausted() {
        let mut t = AllowedLatenessTracker::new();
        t.register("w1", 1_000, 500);
        assert!(t.is_open("w1", 1_400));
        assert!(!t.is_open("w1", 1_500));
        assert!(!t.is_open("w1", 2_000));
    }

    #[test]
    fn allowed_lateness_evict_closed() {
        let mut t = AllowedLatenessTracker::new();
        t.register("a", 100, 100);
        t.register("b", 1_000, 100);
        let evicted = t.evict_closed(500);
        assert!(evicted.contains(&"a".to_string()));
        assert!(!evicted.contains(&"b".to_string()));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn side_output_router_pushes_and_drains() {
        let mut router: SideOutputRouter<i32> = SideOutputRouter::new();
        router.push("late", 1);
        router.push("late", 2);
        router.push("dlq", 7);
        assert_eq!(router.len("late"), 2);
        assert_eq!(router.len("dlq"), 1);
        let drained = router.drain("late");
        assert_eq!(drained, vec![1, 2]);
        assert!(router.is_empty("late"));
        assert_eq!(router.len("dlq"), 1);
    }

    #[test]
    fn re_export_late_data_handler_is_visible() {
        let mut h = LateDataHandler::new(LateDataPolicy::Drop);
        let d = h.handle(10, 50);
        assert_eq!(d, LateDataDecision::Drop);
    }
}
