//! Multi-tenancy support for OxiRS vector search
//!
//! This module provides comprehensive multi-tenancy capabilities including:
//! - Tenant isolation and namespace management
//! - Resource quotas and rate limiting
//! - Usage metering and billing
//! - Access control and authentication
//! - Performance isolation
//! - SLA-based resource allocation with token-bucket admission control

pub mod access_control;
pub mod admission_controller;
pub mod billing;
pub mod isolation;
pub mod manager;
pub mod priority_queue;
pub mod quota;
pub mod sla;
pub mod tenant;
pub mod types;

pub use access_control::{AccessControl, AccessPolicy, Permission, Role};
pub use admission_controller::{AdmissionController, AdmissionError};
pub use billing::{BillingEngine, BillingMetrics, BillingPeriod, PricingModel, UsageRecord};
pub use isolation::{IsolationLevel, IsolationStrategy, NamespaceManager};
pub use manager::{MultiTenantManager, TenantConfig, TenantManagerConfig};
pub use priority_queue::{PrioritizedQuery, PriorityDispatcher, SlaQueryDispatcher};
pub use quota::{QuotaEnforcer, QuotaLimits, QuotaUsage, RateLimiter, ResourceQuota, ResourceType};
pub use sla::{SlaClass, SlaThresholds};
pub use tenant::{Tenant, TenantId, TenantMetadata, TenantStatus};
pub use types::{
    MultiTenancyError, MultiTenancyResult, TenantContext, TenantOperation, TenantStatistics,
};
