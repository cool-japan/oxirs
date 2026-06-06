//! Multi-Tenant Isolation for OxiRS TDB
//!
//! Provides per-tenant namespace isolation, quota enforcement, and
//! cross-tenant access auditing on top of the TDB storage engine.
//!
//! ## Design
//! - Each tenant gets a unique [`TenantId`] that prefixes all triple storage keys.
//! - [`TenantStore`] wraps a TDB store and transparently namespaces all operations.
//! - [`TenantRegistry`] manages tenant lifecycle (create, delete, list).
//! - [`TenantAuditLog`] records cross-tenant access attempts.
//! - Quotas are enforced at write time: inserts fail when limits are exceeded.
//!
//! ## Usage
//! ```rust,no_run
//! use oxirs_tdb::tenant::{TenantId, TenantConfig, TenantRegistry, TenantStore};
//!
//! let mut registry = TenantRegistry::new();
//! let id = TenantId::new("acme_corp").unwrap();
//! let config = TenantConfig {
//!     max_triples: 1_000_000,
//!     max_graphs: 100,
//!     quota_bytes: 512 * 1024 * 1024,
//!     allowed_predicates: vec![],
//!     allowed_prefixes: vec![],
//!     active: true,
//! };
//! registry.create_tenant(id, config).unwrap();
//! ```

pub(crate) mod isolation;
/// Tenant lifecycle management.
pub mod registry;
#[cfg(test)]
mod tests;
/// Type definitions, errors and audit log.
pub mod types;

pub use self::isolation::*;
pub use self::registry::*;
pub use self::types::*;
