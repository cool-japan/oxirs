//! # Per-stream SLA admission control
//!
//! Wraps the shared `oxirs_core::sla` primitives in stream-aware abstractions:
//!
//! * [`StreamSlaConfig`] — per-stream operator-pipeline SLA contract
//!   (`max_events_per_sec`, `max_lag`, `jitter_budget`).
//! * [`admission::StreamAdmissionController`] — token-bucket admission keyed
//!   on `stream_id` and built on top of
//!   [`oxirs_core::sla::AdmissionController`].
//! * [`backpressure_integration::SlaBackpressureCoordinator`] — fuses SLA
//!   admission with the adaptive load shedder so admission decisions take
//!   precedence over best-effort load shedding.
//!
//! Re-exports
//!
//! ```ignore
//! use oxirs_stream::sla::{StreamSlaConfig, StreamAdmissionController};
//! use oxirs_core::sla::SlaClass;
//!
//! let mut ctrl = StreamAdmissionController::new();
//! ctrl.register_stream("orders", StreamSlaConfig::for_class(SlaClass::Gold));
//! ```

pub mod admission;
pub mod backpressure_integration;

pub use admission::{
    StreamAdmissionController, StreamAdmissionDecision, StreamAdmissionStats, StreamSlaConfig,
};
pub use backpressure_integration::{
    BackpressureAction, SlaBackpressureCoordinator, SlaBackpressureDecision, SlaBackpressurePolicy,
};
