//! # Watermark-aware Window Operators
//!
//! This module hosts the *watermark-aware* window operators that complement
//! the time-based windows already implemented in
//! [`crate::processing::window`].  The latter offers tumbling / sliding /
//! session windows driven by wall-clock time; the operators here are driven
//! by event-time watermarks and therefore close their state deterministically
//! when the input watermark advances past `window_end + allowed_lateness`.
//!
//! ## Submodules
//!
//! * [`joins`] — three watermark-driven window-join semantics:
//!   * tumbling-tumbling
//!   * tumbling-sliding
//!   * session-session
//!
//! All joins use the shared [`crate::watermark::propagation::WatermarkPropagator`]
//! contract: at every event admission the operator advances its internal
//! watermark from the *minimum* of upstream watermarks, then emits/cleans up
//! windows whose end time + allowed-lateness budget is below the watermark.

pub mod joins;

pub use joins::{
    SessionSessionJoin, SessionSessionJoinConfig, TumblingSlidingJoin, TumblingSlidingJoinConfig,
    TumblingTumblingJoin, TumblingTumblingJoinConfig, WindowJoinKey, WindowJoinResult,
    WindowJoinStats,
};
