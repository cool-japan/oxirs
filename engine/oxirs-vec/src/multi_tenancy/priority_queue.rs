//! Compatibility shim — priority dispatcher has moved to
//! [`oxirs_core::sla::priority_dispatcher`].
//!
//! The historical name `SlaQueryDispatcher` is preserved as a type alias so
//! existing `oxirs-vec` call sites keep compiling.

pub use oxirs_core::sla::{PrioritizedQuery, PriorityDispatcher};

/// Backwards-compatible alias for [`oxirs_core::sla::PriorityDispatcher`].
///
/// New code should prefer [`PriorityDispatcher`] directly.
pub type SlaQueryDispatcher<T> = PriorityDispatcher<T>;
