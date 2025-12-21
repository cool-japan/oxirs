//! Resource quota and rate limiting for multi-tenancy

use crate::multi_tenancy::types::{MultiTenancyError, MultiTenancyResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Resource types that can be limited
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// Total number of vectors stored
    VectorCount,

    /// Storage space in bytes
    StorageBytes,

    /// Memory usage in bytes
    MemoryBytes,

    /// API calls per time period
    ApiCalls,

    /// Queries per second
    QueriesPerSecond,

    /// Index builds per day
    IndexBuilds,

    /// Embedding generations per day
    EmbeddingGenerations,

    /// Concurrent requests
    ConcurrentRequests,

    /// Batch size
    BatchSize,

    /// Custom resource type
    Custom(u32),
}

impl ResourceType {
    /// Get resource name
    pub fn name(&self) -> String {
        match self {
            Self::VectorCount => "vector_count".to_string(),
            Self::StorageBytes => "storage_bytes".to_string(),
            Self::MemoryBytes => "memory_bytes".to_string(),
            Self::ApiCalls => "api_calls".to_string(),
            Self::QueriesPerSecond => "queries_per_second".to_string(),
            Self::IndexBuilds => "index_builds".to_string(),
            Self::EmbeddingGenerations => "embedding_generations".to_string(),
            Self::ConcurrentRequests => "concurrent_requests".to_string(),
            Self::BatchSize => "batch_size".to_string(),
            Self::Custom(id) => format!("custom_{}", id),
        }
    }
}

/// Resource quota definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuota {
    /// Resource type
    pub resource_type: ResourceType,

    /// Maximum allowed value (None = unlimited)
    pub limit: Option<u64>,

    /// Soft limit for warnings
    pub soft_limit: Option<u64>,

    /// Time window for rate-based limits (seconds)
    pub time_window_secs: Option<u64>,
}

impl ResourceQuota {
    /// Create a new quota with hard limit
    pub fn new(resource_type: ResourceType, limit: u64) -> Self {
        Self {
            resource_type,
            limit: Some(limit),
            soft_limit: None,
            time_window_secs: None,
        }
    }

    /// Create unlimited quota
    pub fn unlimited(resource_type: ResourceType) -> Self {
        Self {
            resource_type,
            limit: None,
            soft_limit: None,
            time_window_secs: None,
        }
    }

    /// Set soft limit
    pub fn with_soft_limit(mut self, soft_limit: u64) -> Self {
        self.soft_limit = Some(soft_limit);
        self
    }

    /// Set time window for rate limits
    pub fn with_time_window(mut self, window_secs: u64) -> Self {
        self.time_window_secs = Some(window_secs);
        self
    }

    /// Check if value exceeds hard limit
    pub fn exceeds_hard_limit(&self, value: u64) -> bool {
        if let Some(limit) = self.limit {
            value > limit
        } else {
            false
        }
    }

    /// Check if value exceeds soft limit
    pub fn exceeds_soft_limit(&self, value: u64) -> bool {
        if let Some(soft_limit) = self.soft_limit {
            value > soft_limit
        } else {
            false
        }
    }
}

/// Collection of quota limits for a tenant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaLimits {
    /// Tenant identifier
    pub tenant_id: String,

    /// Resource quotas
    pub quotas: HashMap<ResourceType, ResourceQuota>,

    /// Whether to enforce quotas strictly
    pub strict_enforcement: bool,
}

impl QuotaLimits {
    /// Create new quota limits for a tenant
    pub fn new(tenant_id: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            quotas: HashMap::new(),
            strict_enforcement: true,
        }
    }

    /// Create default quotas for free tier
    pub fn free_tier(tenant_id: impl Into<String>) -> Self {
        let mut limits = Self::new(tenant_id);
        limits.set_quota(ResourceQuota::new(ResourceType::VectorCount, 10_000));
        limits.set_quota(ResourceQuota::new(ResourceType::StorageBytes, 100_000_000)); // 100MB
        limits.set_quota(ResourceQuota::new(ResourceType::ApiCalls, 1000).with_time_window(3600));
        limits.set_quota(ResourceQuota::new(ResourceType::QueriesPerSecond, 10));
        limits
    }

    /// Create default quotas for pro tier
    pub fn pro_tier(tenant_id: impl Into<String>) -> Self {
        let mut limits = Self::new(tenant_id);
        limits.set_quota(ResourceQuota::new(ResourceType::VectorCount, 1_000_000));
        limits.set_quota(ResourceQuota::new(
            ResourceType::StorageBytes,
            10_000_000_000,
        )); // 10GB
        limits
            .set_quota(ResourceQuota::new(ResourceType::ApiCalls, 100_000).with_time_window(3600));
        limits.set_quota(ResourceQuota::new(ResourceType::QueriesPerSecond, 100));
        limits
    }

    /// Create unlimited quotas for enterprise tier
    pub fn enterprise_tier(tenant_id: impl Into<String>) -> Self {
        let mut limits = Self::new(tenant_id);
        limits.set_quota(ResourceQuota::unlimited(ResourceType::VectorCount));
        limits.set_quota(ResourceQuota::unlimited(ResourceType::StorageBytes));
        limits.set_quota(ResourceQuota::unlimited(ResourceType::ApiCalls));
        limits.set_quota(ResourceQuota::unlimited(ResourceType::QueriesPerSecond));
        limits
    }

    /// Set a quota
    pub fn set_quota(&mut self, quota: ResourceQuota) {
        self.quotas.insert(quota.resource_type, quota);
    }

    /// Get a quota
    pub fn get_quota(&self, resource_type: ResourceType) -> Option<&ResourceQuota> {
        self.quotas.get(&resource_type)
    }

    /// Check if resource usage is within limits
    pub fn check_limit(&self, resource_type: ResourceType, value: u64) -> bool {
        if let Some(quota) = self.get_quota(resource_type) {
            !quota.exceeds_hard_limit(value)
        } else {
            true // No quota defined = unlimited
        }
    }
}

/// Current resource usage for a tenant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaUsage {
    /// Tenant identifier
    pub tenant_id: String,

    /// Current usage by resource type
    pub usage: HashMap<ResourceType, u64>,

    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
}

impl QuotaUsage {
    /// Create new usage tracking
    pub fn new(tenant_id: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            usage: HashMap::new(),
            updated_at: Utc::now(),
        }
    }

    /// Get current usage for a resource
    pub fn get(&self, resource_type: ResourceType) -> u64 {
        *self.usage.get(&resource_type).unwrap_or(&0)
    }

    /// Set usage for a resource
    pub fn set(&mut self, resource_type: ResourceType, value: u64) {
        self.usage.insert(resource_type, value);
        self.updated_at = Utc::now();
    }

    /// Increment usage
    pub fn increment(&mut self, resource_type: ResourceType, amount: u64) {
        let current = self.get(resource_type);
        self.set(resource_type, current + amount);
    }

    /// Decrement usage
    pub fn decrement(&mut self, resource_type: ResourceType, amount: u64) {
        let current = self.get(resource_type);
        if current >= amount {
            self.set(resource_type, current - amount);
        } else {
            self.set(resource_type, 0);
        }
    }

    /// Reset usage for a resource
    pub fn reset(&mut self, resource_type: ResourceType) {
        self.set(resource_type, 0);
    }

    /// Reset all usage
    pub fn reset_all(&mut self) {
        self.usage.clear();
        self.updated_at = Utc::now();
    }
}

/// Quota enforcement engine
pub struct QuotaEnforcer {
    /// Quota limits by tenant
    limits: Arc<Mutex<HashMap<String, QuotaLimits>>>,

    /// Current usage by tenant
    usage: Arc<Mutex<HashMap<String, QuotaUsage>>>,
}

impl QuotaEnforcer {
    /// Create new quota enforcer
    pub fn new() -> Self {
        Self {
            limits: Arc::new(Mutex::new(HashMap::new())),
            usage: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Set quota limits for a tenant
    pub fn set_limits(&self, limits: QuotaLimits) -> MultiTenancyResult<()> {
        let tenant_id = limits.tenant_id.clone();
        self.limits
            .lock()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?
            .insert(tenant_id.clone(), limits);

        // Initialize usage if not exists
        let mut usage_map = self
            .usage
            .lock()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?;
        usage_map
            .entry(tenant_id.clone())
            .or_insert_with(|| QuotaUsage::new(tenant_id));

        Ok(())
    }

    /// Check if tenant can consume resource
    pub fn check_quota(
        &self,
        tenant_id: &str,
        resource_type: ResourceType,
        amount: u64,
    ) -> MultiTenancyResult<bool> {
        let limits = self
            .limits
            .lock()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?;

        let usage = self
            .usage
            .lock()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?;

        if let Some(tenant_limits) = limits.get(tenant_id) {
            if let Some(tenant_usage) = usage.get(tenant_id) {
                let current = tenant_usage.get(resource_type);
                let new_usage = current + amount;

                return Ok(tenant_limits.check_limit(resource_type, new_usage));
            }
        }

        // No limits defined = allow
        Ok(true)
    }

    /// Consume resource quota
    pub fn consume(
        &self,
        tenant_id: &str,
        resource_type: ResourceType,
        amount: u64,
    ) -> MultiTenancyResult<()> {
        // Check quota first
        if !self.check_quota(tenant_id, resource_type, amount)? {
            return Err(MultiTenancyError::QuotaExceeded {
                tenant_id: tenant_id.to_string(),
                resource: resource_type.name(),
            });
        }

        // Increment usage
        let mut usage = self
            .usage
            .lock()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?;

        usage
            .entry(tenant_id.to_string())
            .or_insert_with(|| QuotaUsage::new(tenant_id))
            .increment(resource_type, amount);

        Ok(())
    }

    /// Release resource quota
    pub fn release(
        &self,
        tenant_id: &str,
        resource_type: ResourceType,
        amount: u64,
    ) -> MultiTenancyResult<()> {
        let mut usage = self
            .usage
            .lock()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?;

        usage
            .entry(tenant_id.to_string())
            .or_insert_with(|| QuotaUsage::new(tenant_id))
            .decrement(resource_type, amount);

        Ok(())
    }

    /// Get current usage for tenant
    pub fn get_usage(&self, tenant_id: &str) -> MultiTenancyResult<QuotaUsage> {
        let usage = self
            .usage
            .lock()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?;

        Ok(usage
            .get(tenant_id)
            .cloned()
            .unwrap_or_else(|| QuotaUsage::new(tenant_id)))
    }

    /// Reset usage for a tenant
    pub fn reset_usage(&self, tenant_id: &str) -> MultiTenancyResult<()> {
        let mut usage = self
            .usage
            .lock()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?;

        if let Some(tenant_usage) = usage.get_mut(tenant_id) {
            tenant_usage.reset_all();
        }

        Ok(())
    }
}

impl Default for QuotaEnforcer {
    fn default() -> Self {
        Self::new()
    }
}

/// Rate limiter for API calls
pub struct RateLimiter {
    /// Token buckets by tenant
    buckets: Arc<Mutex<HashMap<String, TokenBucket>>>,
}

/// Token bucket for rate limiting
#[derive(Debug, Clone)]
struct TokenBucket {
    /// Current number of tokens
    tokens: f64,

    /// Maximum capacity
    capacity: f64,

    /// Refill rate (tokens per second)
    refill_rate: f64,

    /// Last refill time
    last_refill: DateTime<Utc>,
}

impl TokenBucket {
    /// Create new token bucket
    fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            tokens: capacity,
            capacity,
            refill_rate,
            last_refill: Utc::now(),
        }
    }

    /// Refill tokens based on elapsed time
    fn refill(&mut self) {
        let now = Utc::now();
        let elapsed = (now - self.last_refill).num_milliseconds() as f64 / 1000.0;
        let new_tokens = elapsed * self.refill_rate;

        self.tokens = (self.tokens + new_tokens).min(self.capacity);
        self.last_refill = now;
    }

    /// Try to consume tokens
    fn try_consume(&mut self, amount: f64) -> bool {
        self.refill();

        if self.tokens >= amount {
            self.tokens -= amount;
            true
        } else {
            false
        }
    }
}

impl RateLimiter {
    /// Create new rate limiter
    pub fn new() -> Self {
        Self {
            buckets: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Set rate limit for a tenant (requests per second)
    pub fn set_rate(
        &self,
        tenant_id: impl Into<String>,
        requests_per_second: f64,
    ) -> MultiTenancyResult<()> {
        let bucket = TokenBucket::new(requests_per_second * 2.0, requests_per_second);

        self.buckets
            .lock()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?
            .insert(tenant_id.into(), bucket);

        Ok(())
    }

    /// Check if request is allowed
    pub fn allow_request(&self, tenant_id: &str) -> MultiTenancyResult<bool> {
        let mut buckets = self
            .buckets
            .lock()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?;

        if let Some(bucket) = buckets.get_mut(tenant_id) {
            Ok(bucket.try_consume(1.0))
        } else {
            // No rate limit configured = allow
            Ok(true)
        }
    }

    /// Check if batch request is allowed
    pub fn allow_batch_request(
        &self,
        tenant_id: &str,
        batch_size: usize,
    ) -> MultiTenancyResult<bool> {
        let mut buckets = self
            .buckets
            .lock()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?;

        if let Some(bucket) = buckets.get_mut(tenant_id) {
            Ok(bucket.try_consume(batch_size as f64))
        } else {
            // No rate limit configured = allow
            Ok(true)
        }
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration as StdDuration;

    #[test]
    fn test_resource_quota() {
        let quota = ResourceQuota::new(ResourceType::VectorCount, 1000);
        assert_eq!(quota.limit, Some(1000));
        assert!(!quota.exceeds_hard_limit(500));
        assert!(quota.exceeds_hard_limit(1001));

        let quota = quota.with_soft_limit(800);
        assert!(!quota.exceeds_soft_limit(700));
        assert!(quota.exceeds_soft_limit(900));
    }

    #[test]
    fn test_quota_limits() {
        let limits = QuotaLimits::free_tier("tenant1");
        assert!(limits.get_quota(ResourceType::VectorCount).is_some());
        assert!(limits.check_limit(ResourceType::VectorCount, 5000));
        assert!(!limits.check_limit(ResourceType::VectorCount, 20000));
    }

    #[test]
    fn test_quota_usage() {
        let mut usage = QuotaUsage::new("tenant1");
        assert_eq!(usage.get(ResourceType::VectorCount), 0);

        usage.increment(ResourceType::VectorCount, 100);
        assert_eq!(usage.get(ResourceType::VectorCount), 100);

        usage.increment(ResourceType::VectorCount, 50);
        assert_eq!(usage.get(ResourceType::VectorCount), 150);

        usage.decrement(ResourceType::VectorCount, 30);
        assert_eq!(usage.get(ResourceType::VectorCount), 120);

        usage.reset(ResourceType::VectorCount);
        assert_eq!(usage.get(ResourceType::VectorCount), 0);
    }

    #[test]
    fn test_quota_enforcer() {
        let enforcer = QuotaEnforcer::new();
        let limits = QuotaLimits::free_tier("tenant1");
        enforcer.set_limits(limits).unwrap();

        // Should allow within limits
        assert!(enforcer
            .check_quota("tenant1", ResourceType::VectorCount, 5000)
            .unwrap());

        // Consume some quota
        enforcer
            .consume("tenant1", ResourceType::VectorCount, 5000)
            .unwrap();

        // Should still allow more
        assert!(enforcer
            .check_quota("tenant1", ResourceType::VectorCount, 3000)
            .unwrap());

        // Should reject exceeding quota
        assert!(!enforcer
            .check_quota("tenant1", ResourceType::VectorCount, 10000)
            .unwrap());

        // Consuming beyond quota should fail
        assert!(enforcer
            .consume("tenant1", ResourceType::VectorCount, 10000)
            .is_err());
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = RateLimiter::new();
        limiter.set_rate("tenant1", 10.0).unwrap(); // 10 requests per second

        // Should allow initial requests
        assert!(limiter.allow_request("tenant1").unwrap());
        assert!(limiter.allow_request("tenant1").unwrap());

        // Allow batch request
        assert!(limiter.allow_batch_request("tenant1", 5).unwrap());

        // After consuming many tokens, should be denied
        for _ in 0..20 {
            let _ = limiter.allow_request("tenant1");
        }
        assert!(!limiter.allow_request("tenant1").unwrap());

        // After waiting, tokens should refill
        thread::sleep(StdDuration::from_millis(200));
        assert!(limiter.allow_request("tenant1").unwrap());
    }

    #[test]
    fn test_tier_quotas() {
        let free = QuotaLimits::free_tier("tenant1");
        let pro = QuotaLimits::pro_tier("tenant2");
        let enterprise = QuotaLimits::enterprise_tier("tenant3");

        // Free tier should have restrictive limits
        assert!(free.check_limit(ResourceType::VectorCount, 5000));
        assert!(!free.check_limit(ResourceType::VectorCount, 20000));

        // Pro tier should have higher limits
        assert!(pro.check_limit(ResourceType::VectorCount, 500000));
        assert!(!pro.check_limit(ResourceType::VectorCount, 2000000));

        // Enterprise should be unlimited
        assert!(enterprise.check_limit(ResourceType::VectorCount, 10000000));
        assert!(enterprise.check_limit(ResourceType::VectorCount, 100000000));
    }
}
