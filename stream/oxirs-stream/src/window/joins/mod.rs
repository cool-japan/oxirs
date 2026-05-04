//! Watermark-driven window joins.
//!
//! Three join semantics are implemented:
//!
//! 1. [`tumbling_tumbling::TumblingTumblingJoin`] — both sides use the same
//!    fixed tumbling window; only events whose timestamps fall in the same
//!    window pane are considered for joining.  Closure: when the watermark
//!    advances past `pane_end + allowed_lateness_ms`.
//!
//! 2. [`tumbling_sliding::TumblingSlidingJoin`] — the *left* stream uses
//!    tumbling windows; the *right* stream uses sliding windows.  An event
//!    on either side is considered against every active right-pane that
//!    overlaps with its tumbling pane.
//!
//! 3. [`session_session::SessionSessionJoin`] — both sides use session
//!    windows defined by an inactivity gap.  Two events join when their
//!    sessions overlap (at least one event timestamp from each side falls
//!    within the union session).
//!
//! All three implementations share:
//!
//! * Per-stream key extraction via a closure (`F: Fn(&L) -> K`).
//! * Deterministic state cleanup driven by an externally supplied watermark.
//! * Configurable allowed lateness measured in milliseconds.
//! * Statistics ([`WindowJoinStats`]) tracking emitted / dropped / late
//!   events.
//!
//! Refer to `docs/engine_overview.md` for the watermark/window/join contract.

pub mod session_session;
pub mod tumbling_sliding;
pub mod tumbling_tumbling;

pub use session_session::{SessionSessionJoin, SessionSessionJoinConfig};
pub use tumbling_sliding::{TumblingSlidingJoin, TumblingSlidingJoinConfig};
pub use tumbling_tumbling::{TumblingTumblingJoin, TumblingTumblingJoinConfig};

// ─── Shared types ────────────────────────────────────────────────────────────

/// Join key extracted from each event.
pub type WindowJoinKey = String;

/// A successful join result.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowJoinResult<L, R> {
    pub key: WindowJoinKey,
    pub left: L,
    pub right: R,
    /// Pane end (or session end) at which this join is emitted.
    pub pane_end_ms: i64,
}

/// Per-join statistics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WindowJoinStats {
    pub left_events: u64,
    pub right_events: u64,
    pub joined_pairs: u64,
    pub late_events_dropped: u64,
    pub windows_closed: u64,
}
