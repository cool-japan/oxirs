//! # Per-node SLA admission control for the cluster
//!
//! Reuses the shared per-tenant SLA primitives from
//! [`oxirs_core::sla`] and wraps them in cluster-aware abstractions:
//!
//! * [`SlaAdmissionConfig`] — per-class quotas (`max_qps_per_class`,
//!   `max_concurrent_per_class`).
//! * [`admission::ClusterAdmissionController`] — token-bucket admission keyed
//!   on [`oxirs_core::sla::SlaClass`] (one synthetic tenant per class on the
//!   shared `AdmissionController`) plus a strict
//!   `max_concurrent_per_class` semaphore-style counter.
//! * [`proposer_gate::SlaProposerGate`] — wraps a Raft log proposer; rejects
//!   writes that exceed the per-class admission budget.
//! * [`reader_gate::SlaReaderGate`] — wraps a read-replica handler; rejects
//!   reads that exceed the per-class admission budget.
//!
//! ## Integration
//!
//! The gates are *additive* — existing call sites continue to work unmodified.
//! New call sites that opt in obtain an [`SlaProposerGate`] (or
//! [`SlaReaderGate`]) and dispatch through it.
//!
//! ## Example
//!
//! ```ignore
//! use oxirs_cluster::sla::{
//!     SlaAdmissionConfig, ClusterAdmissionController, SlaProposerGate,
//! };
//! use oxirs_core::sla::SlaClass;
//! use std::sync::Arc;
//!
//! let cfg = SlaAdmissionConfig::default();
//! let controller = Arc::new(ClusterAdmissionController::new(cfg));
//! controller.register_class(SlaClass::Bronze);
//! controller.register_class(SlaClass::Gold);
//!
//! let gate = SlaProposerGate::new(controller.clone());
//! gate.try_acquire(SlaClass::Bronze)?;
//! // ... do the actual Raft propose ...
//! gate.release(SlaClass::Bronze);
//! # Ok::<(), oxirs_cluster::sla::SlaError>(())
//! ```

pub mod admission;
pub mod proposer_gate;
pub mod reader_gate;

pub use admission::{
    ClusterAdmissionController, ClusterAdmissionStats, SlaAdmissionConfig, SlaClassQuota, SlaError,
};
pub use proposer_gate::{ProposerOutcome, SlaProposerGate};
pub use reader_gate::{ReaderOutcome, SlaReaderGate};
