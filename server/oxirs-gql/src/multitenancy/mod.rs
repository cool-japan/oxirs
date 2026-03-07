//! Multi-tenant support for GraphQL over RDF
//!
//! This module provides per-tenant schema isolation, dataset access control,
//! and request-scoped tenant context propagation.

pub mod tenant_registry;

pub use tenant_registry::{
    TenantConfig, TenantContext, TenantCustomType, TenantField, TenantOperation, TenantRegistry,
};
