//! Multi-tenant GraphQL support for OxiRS.
//!
//! This module extends the basic `multitenancy` infrastructure with:
//! - Per-tenant GraphQL schema management (`TenantSchemaRegistry`)
//! - Query filtering based on tenant access policies (`TenantQueryFilter`)
//! - Per-tenant rate limiting (`TenantRateLimiter`, `TenantLimits`)

pub mod query_filter;
pub mod rate_limiter;
pub mod schema_registry;

pub use query_filter::{AccessViolation, TenantQueryFilter};
pub use rate_limiter::{RateLimitResult, TenantLimits, TenantRateLimiter};
pub use schema_registry::{TenantSchema, TenantSchemaRegistry};
