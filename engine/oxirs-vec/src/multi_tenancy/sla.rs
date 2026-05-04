//! Compatibility shim — SLA primitives now live in [`oxirs_core::sla`].
//!
//! This module preserves the old `oxirs_vec::multi_tenancy::sla::*` paths so
//! existing call sites keep compiling.  All real implementation has moved into
//! [`oxirs_core::sla`] in W2-S4 to enable shared admission control across
//! `oxirs-arq`, `oxirs-vec`, `oxirs-cluster`, and `oxirs-stream`.

pub use oxirs_core::sla::{SlaClass, SlaThresholds};
