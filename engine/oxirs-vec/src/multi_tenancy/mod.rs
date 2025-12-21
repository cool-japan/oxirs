//! Multi-tenancy support for OxiRS vector search
//!
//! This module provides comprehensive multi-tenancy capabilities including:
//! - Tenant isolation and namespace management
//! - Resource quotas and rate limiting
//! - Usage metering and billing
//! - Access control and authentication
//! - Performance isolation

pub mod access_control;
pub mod billing;
pub mod isolation;
pub mod manager;
pub mod quota;
pub mod tenant;
pub mod types;

pub use access_control::{AccessControl, AccessPolicy, Permission, Role};
pub use billing::{BillingEngine, BillingMetrics, BillingPeriod, PricingModel, UsageRecord};
pub use isolation::{IsolationLevel, IsolationStrategy, NamespaceManager};
pub use manager::{MultiTenantManager, TenantConfig, TenantManagerConfig};
pub use quota::{QuotaEnforcer, QuotaLimits, QuotaUsage, RateLimiter, ResourceQuota, ResourceType};
pub use tenant::{Tenant, TenantId, TenantMetadata, TenantStatus};
pub use types::{
    MultiTenancyError, MultiTenancyResult, TenantContext, TenantOperation, TenantStatistics,
};
