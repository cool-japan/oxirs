//! Service Registry and Management
//!
//! This module manages federated services, their capabilities, and health status.
//!
//! This is a thin facade that re-exports the public API from cohesive sibling
//! modules:
//! - [`crate::service_types`] — service types, capabilities, auth, status,
//!   metadata, performance, and connection pool descriptors.
//! - [`crate::service_core`] — the [`FederatedServiceRegistry`] implementation.

pub use crate::service_core::*;
pub use crate::service_types::*;
