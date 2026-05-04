//! Shared per-tenant SLA primitives for the OxiRS ecosystem.
//!
//! This module defines four service tiers — Bronze → Silver → Gold → Platinum
//! — each with associated latency, concurrency, and token-bucket parameters.
//! Higher tiers receive more tokens, tighter latency budgets, and higher
//! dispatch priority.
//!
//! It also provides:
//!
//! * [`AdmissionController`]: a thread-safe token-bucket admission gate keyed
//!   on `(tenant_id, SlaClass)`.
//! * [`PriorityDispatcher`]: a max-heap dispatcher that always returns the
//!   highest-priority queued payload first, breaking ties by FIFO order.
//!
//! These primitives are consumed by:
//!
//! * `oxirs-arq` for SPARQL query admission control,
//! * `oxirs-vec` for vector-search admission control (re-exported via
//!   `oxirs_vec::multi_tenancy`),
//! * `oxirs-cluster` / `oxirs-stream` for distributed admission control
//!   (W2-S5/W2-S6 and beyond).

pub mod admission_controller;
pub mod class;
pub mod priority_dispatcher;
pub mod thresholds;

pub use admission_controller::{AdmissionController, AdmissionError};
pub use class::SlaClass;
pub use priority_dispatcher::{PrioritizedQuery, PriorityDispatcher};
pub use thresholds::SlaThresholds;
