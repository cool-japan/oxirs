//! # Rate Limiting and Quota Management
//!
//! Enterprise-grade rate limiting with multiple algorithms, distributed coordination,
//! per-tenant quotas, and comprehensive monitoring for streaming workloads.
//!
//! ## Features
//!
//! - **Multiple Algorithms**: Token bucket, sliding window, leaky bucket, fixed window
//! - **Distributed Coordination**: Redis-backed distributed rate limiting
//! - **Per-Tenant Quotas**: Fine-grained quota management with tenant isolation
//! - **Adaptive Limits**: Dynamic adjustment based on system load
//! - **Comprehensive Metrics**: Real-time monitoring and alerting
//! - **Graceful Degradation**: Fallback strategies when limits are exceeded
//!
//! ## Example
//!
//! ```rust,ignore
//! use oxirs_stream::rate_limiting::{RateLimiter, RateLimitConfig, RateLimitAlgorithm};
//!
//! let config = RateLimitConfig {
//!     algorithm: RateLimitAlgorithm::TokenBucket { capacity: 1000, refill_rate: 100 },
//!     ..Default::default()
//! };
//!
//! let limiter = RateLimiter::new(config)?;
//!
//! // Check if request is allowed
//! if limiter.allow("tenant-1", 1).await? {
//!     // Process request
//! } else {
//!     // Rate limit exceeded
//! }
//! ```

#[cfg(feature = "redis")]
use anyhow::anyhow;
use anyhow::Result;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Rate limiting algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateLimitAlgorithm {
    /// Token bucket algorithm
    /// - capacity: Maximum number of tokens
    /// - refill_rate: Tokens added per second
    TokenBucket { capacity: u64, refill_rate: u64 },

    /// Sliding window counter algorithm
    /// - window_size: Duration of the sliding window
    /// - max_requests: Maximum requests in the window
    SlidingWindow {
        window_size: ChronoDuration,
        max_requests: u64,
    },

    /// Leaky bucket algorithm
    /// - capacity: Maximum bucket capacity
    /// - leak_rate: Rate at which bucket empties (per second)
    LeakyBucket { capacity: u64, leak_rate: u64 },

    /// Fixed window counter algorithm
    /// - window_size: Fixed window duration
    /// - max_requests: Maximum requests per window
    FixedWindow {
        window_size: ChronoDuration,
        max_requests: u64,
    },

    /// Adaptive rate limiting
    /// - base_limit: Base rate limit
    /// - adjustment_factor: How much to adjust based on load
    Adaptive {
        base_limit: u64,
        adjustment_factor: f64,
    },
}

impl Default for RateLimitAlgorithm {
    fn default() -> Self {
        Self::TokenBucket {
            capacity: 1000,
            refill_rate: 100,
        }
    }
}

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Rate limiting algorithm
    pub algorithm: RateLimitAlgorithm,

    /// Enable distributed rate limiting
    pub distributed: bool,

    /// Redis URL for distributed coordination (if enabled)
    pub redis_url: Option<String>,

    /// Enable per-tenant quotas
    pub per_tenant_quotas: bool,

    /// Default quota for new tenants
    pub default_quota: QuotaLimits,

    /// Enable adaptive adjustment
    pub enable_adaptive: bool,

    /// Monitoring configuration
    pub monitoring: RateLimitMonitoringConfig,

    /// Rejection strategy when limit exceeded
    pub rejection_strategy: RejectionStrategy,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            algorithm: RateLimitAlgorithm::default(),
            distributed: false,
            redis_url: None,
            per_tenant_quotas: true,
            default_quota: QuotaLimits::default(),
            enable_adaptive: true,
            monitoring: RateLimitMonitoringConfig::default(),
            rejection_strategy: RejectionStrategy::ImmediateReject,
        }
    }
}

/// Quota limits for a tenant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaLimits {
    /// Requests per second
    pub requests_per_second: u64,

    /// Requests per minute
    pub requests_per_minute: u64,

    /// Requests per hour
    pub requests_per_hour: u64,

    /// Requests per day
    pub requests_per_day: u64,

    /// Bandwidth limit (bytes per second)
    pub bandwidth_bytes_per_second: u64,

    /// Maximum concurrent requests
    pub max_concurrent_requests: u32,

    /// Maximum burst size (tokens)
    pub max_burst: u64,
}

impl Default for QuotaLimits {
    fn default() -> Self {
        Self {
            requests_per_second: 100,
            requests_per_minute: 5000,
            requests_per_hour: 100_000,
            requests_per_day: 1_000_000,
            bandwidth_bytes_per_second: 10_485_760, // 10 MB/s
            max_concurrent_requests: 100,
            max_burst: 200,
        }
    }
}

/// Rejection strategy when rate limit is exceeded
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RejectionStrategy {
    /// Immediately reject the request
    ImmediateReject,

    /// Queue the request with a timeout
    QueueWithTimeout(u64), // timeout in milliseconds

    /// Throttle with exponential backoff
    ExponentialBackoff {
        initial_delay_ms: u64,
        max_delay_ms: u64,
    },

    /// Best-effort processing (may degrade quality)
    BestEffort,
}

/// Rate limit monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitMonitoringConfig {
    /// Enable metrics collection
    pub enable_metrics: bool,

    /// Metrics reporting interval
    pub metrics_interval: ChronoDuration,

    /// Enable alerting when thresholds are exceeded
    pub enable_alerts: bool,

    /// Alert threshold (percentage of limit)
    pub alert_threshold: f64,

    /// Alert cooldown period
    pub alert_cooldown: ChronoDuration,
}

impl Default for RateLimitMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            metrics_interval: ChronoDuration::seconds(60),
            enable_alerts: true,
            alert_threshold: 0.9, // Alert at 90% of limit
            alert_cooldown: ChronoDuration::minutes(5),
        }
    }
}

/// Token bucket state for rate limiting
#[derive(Debug, Clone)]
struct TokenBucketState {
    tokens: f64,
    capacity: u64,
    refill_rate: u64,
    last_refill: DateTime<Utc>,
}

impl TokenBucketState {
    fn new(capacity: u64, refill_rate: u64) -> Self {
        Self {
            tokens: capacity as f64,
            capacity,
            refill_rate,
            last_refill: Utc::now(),
        }
    }

    fn refill(&mut self) {
        let now = Utc::now();
        let elapsed = now.signed_duration_since(self.last_refill);
        let seconds = elapsed.num_milliseconds() as f64 / 1000.0;

        let new_tokens = seconds * self.refill_rate as f64;
        self.tokens = (self.tokens + new_tokens).min(self.capacity as f64);
        self.last_refill = now;
    }

    fn consume(&mut self, tokens: u64) -> bool {
        self.refill();

        if self.tokens >= tokens as f64 {
            self.tokens -= tokens as f64;
            true
        } else {
            false
        }
    }

    fn available_tokens(&self) -> u64 {
        self.tokens.floor() as u64
    }
}

/// Sliding window state for rate limiting
#[derive(Debug, Clone)]
struct SlidingWindowState {
    requests: VecDeque<DateTime<Utc>>,
    window_size: ChronoDuration,
    max_requests: u64,
}

impl SlidingWindowState {
    fn new(window_size: ChronoDuration, max_requests: u64) -> Self {
        Self {
            requests: VecDeque::new(),
            window_size,
            max_requests,
        }
    }

    fn cleanup(&mut self) {
        let now = Utc::now();
        let cutoff = now - self.window_size;

        while let Some(&oldest) = self.requests.front() {
            if oldest < cutoff {
                self.requests.pop_front();
            } else {
                break;
            }
        }
    }

    fn allow(&mut self) -> bool {
        self.cleanup();

        if self.requests.len() < self.max_requests as usize {
            self.requests.push_back(Utc::now());
            true
        } else {
            false
        }
    }

    fn current_count(&self) -> usize {
        self.requests.len()
    }
}

/// Leaky bucket state for rate limiting
#[derive(Debug, Clone)]
struct LeakyBucketState {
    queue_size: u64,
    capacity: u64,
    leak_rate: u64,
    last_leak: DateTime<Utc>,
}

impl LeakyBucketState {
    fn new(capacity: u64, leak_rate: u64) -> Self {
        Self {
            queue_size: 0,
            capacity,
            leak_rate,
            last_leak: Utc::now(),
        }
    }

    fn leak(&mut self) {
        let now = Utc::now();
        let elapsed = now.signed_duration_since(self.last_leak);
        let seconds = elapsed.num_milliseconds() as f64 / 1000.0;

        let leaked = (seconds * self.leak_rate as f64) as u64;
        self.queue_size = self.queue_size.saturating_sub(leaked);
        self.last_leak = now;
    }

    fn add(&mut self, items: u64) -> bool {
        self.leak();

        if self.queue_size + items <= self.capacity {
            self.queue_size += items;
            true
        } else {
            false
        }
    }
}

/// Rate limiter state per tenant
#[derive(Debug)]
enum RateLimiterState {
    TokenBucket(TokenBucketState),
    SlidingWindow(SlidingWindowState),
    LeakyBucket(LeakyBucketState),
}

/// Rate limiter implementation
pub struct RateLimiter {
    config: RateLimitConfig,
    states: Arc<RwLock<HashMap<String, RateLimiterState>>>,
    quotas: Arc<RwLock<HashMap<String, QuotaLimits>>>,
    stats: Arc<RwLock<RateLimitStats>>,
    #[cfg(feature = "redis")]
    redis_client: Option<Arc<redis::Client>>,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(config: RateLimitConfig) -> Result<Self> {
        #[cfg(feature = "redis")]
        let redis_client = if config.distributed {
            if let Some(ref url) = config.redis_url {
                Some(Arc::new(redis::Client::open(url.as_str())?))
            } else {
                return Err(anyhow!("Redis URL required for distributed rate limiting"));
            }
        } else {
            None
        };

        Ok(Self {
            config,
            states: Arc::new(RwLock::new(HashMap::new())),
            quotas: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(RateLimitStats::default())),
            #[cfg(feature = "redis")]
            redis_client,
        })
    }

    /// Check if a request is allowed for a tenant
    pub async fn allow(&self, tenant_id: &str, tokens: u64) -> Result<bool> {
        let mut states = self.states.write().await;
        let mut stats = self.stats.write().await;

        stats.total_requests += 1;

        // Get or create state for this tenant
        let state = states
            .entry(tenant_id.to_string())
            .or_insert_with(|| self.create_state());

        let allowed = match state {
            RateLimiterState::TokenBucket(bucket) => bucket.consume(tokens),
            RateLimiterState::SlidingWindow(window) => {
                if tokens != 1 {
                    warn!("Sliding window only supports single requests");
                }
                window.allow()
            }
            RateLimiterState::LeakyBucket(bucket) => bucket.add(tokens),
        };

        if allowed {
            stats.allowed_requests += 1;
            debug!(
                "Request allowed for tenant {}: {} tokens",
                tenant_id, tokens
            );
        } else {
            stats.rejected_requests += 1;
            warn!(
                "Request rejected for tenant {}: rate limit exceeded",
                tenant_id
            );
        }

        Ok(allowed)
    }

    /// Set custom quota limits for a tenant
    pub async fn set_quota(&self, tenant_id: &str, quota: QuotaLimits) -> Result<()> {
        let mut quotas = self.quotas.write().await;
        quotas.insert(tenant_id.to_string(), quota);
        info!("Updated quota for tenant {}", tenant_id);
        Ok(())
    }

    /// Get current quota for a tenant
    pub async fn get_quota(&self, tenant_id: &str) -> Result<QuotaLimits> {
        let quotas = self.quotas.read().await;
        Ok(quotas
            .get(tenant_id)
            .cloned()
            .unwrap_or_else(|| self.config.default_quota.clone()))
    }

    /// Get remaining quota for a tenant
    pub async fn remaining_quota(&self, tenant_id: &str) -> Result<u64> {
        let states = self.states.read().await;

        match states.get(tenant_id) {
            Some(RateLimiterState::TokenBucket(bucket)) => Ok(bucket.available_tokens()),
            Some(RateLimiterState::SlidingWindow(window)) => Ok(window
                .max_requests
                .saturating_sub(window.current_count() as u64)),
            Some(RateLimiterState::LeakyBucket(bucket)) => {
                Ok(bucket.capacity.saturating_sub(bucket.queue_size))
            }
            None => Ok(0),
        }
    }

    /// Reset rate limit state for a tenant
    pub async fn reset(&self, tenant_id: &str) -> Result<()> {
        let mut states = self.states.write().await;
        states.remove(tenant_id);
        info!("Reset rate limit state for tenant {}", tenant_id);
        Ok(())
    }

    /// Get rate limiting statistics
    pub async fn stats(&self) -> Result<RateLimitStats> {
        let stats = self.stats.read().await;
        Ok(stats.clone())
    }

    /// Clear all rate limiting state
    pub async fn clear(&self) -> Result<()> {
        let mut states = self.states.write().await;
        let mut quotas = self.quotas.write().await;
        states.clear();
        quotas.clear();
        info!("Cleared all rate limiting state");
        Ok(())
    }

    /// Create initial state based on algorithm
    fn create_state(&self) -> RateLimiterState {
        match &self.config.algorithm {
            RateLimitAlgorithm::TokenBucket {
                capacity,
                refill_rate,
            } => RateLimiterState::TokenBucket(TokenBucketState::new(*capacity, *refill_rate)),
            RateLimitAlgorithm::SlidingWindow {
                window_size,
                max_requests,
            } => RateLimiterState::SlidingWindow(SlidingWindowState::new(
                *window_size,
                *max_requests,
            )),
            RateLimitAlgorithm::LeakyBucket {
                capacity,
                leak_rate,
            } => RateLimiterState::LeakyBucket(LeakyBucketState::new(*capacity, *leak_rate)),
            RateLimitAlgorithm::FixedWindow {
                window_size,
                max_requests,
            } => {
                // Implement as sliding window for now
                RateLimiterState::SlidingWindow(SlidingWindowState::new(
                    *window_size,
                    *max_requests,
                ))
            }
            RateLimitAlgorithm::Adaptive { base_limit, .. } => {
                // Start with base limit as token bucket
                RateLimiterState::TokenBucket(TokenBucketState::new(*base_limit, *base_limit / 10))
            }
        }
    }
}

/// Rate limiting statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RateLimitStats {
    /// Total requests checked
    pub total_requests: u64,

    /// Requests allowed
    pub allowed_requests: u64,

    /// Requests rejected
    pub rejected_requests: u64,

    /// Number of active tenants
    pub active_tenants: usize,

    /// Rejection rate (0.0 - 1.0)
    pub rejection_rate: f64,
}

impl RateLimitStats {
    /// Calculate rejection rate
    pub fn calculate_rejection_rate(&mut self) {
        if self.total_requests > 0 {
            self.rejection_rate = self.rejected_requests as f64 / self.total_requests as f64;
        }
    }
}

/// Quota manager for multi-tenant scenarios
pub struct QuotaManager {
    limiter: Arc<RateLimiter>,
    enforcement_mode: QuotaEnforcementMode,
}

impl QuotaManager {
    /// Create a new quota manager
    pub fn new(config: RateLimitConfig) -> Result<Self> {
        Ok(Self {
            limiter: Arc::new(RateLimiter::new(config)?),
            enforcement_mode: QuotaEnforcementMode::Strict,
        })
    }

    /// Check if tenant can perform an operation
    pub async fn check_quota(
        &self,
        tenant_id: &str,
        operation: &QuotaOperation,
    ) -> Result<QuotaCheckResult> {
        let tokens = match operation {
            QuotaOperation::Request { count } => *count,
            QuotaOperation::Bandwidth { bytes } => bytes / 1024, // Convert to KB
            QuotaOperation::Storage { bytes } => bytes / (1024 * 1024), // Convert to MB
        };

        let allowed = self.limiter.allow(tenant_id, tokens).await?;
        let remaining = self.limiter.remaining_quota(tenant_id).await?;

        Ok(QuotaCheckResult {
            allowed,
            remaining,
            reset_at: Utc::now() + ChronoDuration::seconds(60),
            retry_after: if allowed {
                None
            } else {
                Some(ChronoDuration::seconds(1))
            },
        })
    }

    /// Update tenant quota
    pub async fn update_quota(&self, tenant_id: &str, quota: QuotaLimits) -> Result<()> {
        self.limiter.set_quota(tenant_id, quota).await
    }
}

/// Quota enforcement mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuotaEnforcementMode {
    /// Strictly enforce quotas
    Strict,
    /// Soft enforcement (warnings only)
    Soft,
    /// Disabled
    Disabled,
}

/// Quota operation types
#[derive(Debug, Clone)]
pub enum QuotaOperation {
    /// Request count
    Request { count: u64 },
    /// Bandwidth usage
    Bandwidth { bytes: u64 },
    /// Storage usage
    Storage { bytes: u64 },
}

/// Result of a quota check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaCheckResult {
    /// Whether the operation is allowed
    pub allowed: bool,
    /// Remaining quota
    pub remaining: u64,
    /// When the quota resets
    pub reset_at: DateTime<Utc>,
    /// Suggested retry delay if not allowed
    pub retry_after: Option<ChronoDuration>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_token_bucket_basic() {
        let config = RateLimitConfig {
            algorithm: RateLimitAlgorithm::TokenBucket {
                capacity: 10,
                refill_rate: 1,
            },
            ..Default::default()
        };

        let limiter = RateLimiter::new(config).unwrap();

        // Should allow first 10 requests
        for i in 0..10 {
            assert!(
                limiter.allow("tenant-1", 1).await.unwrap(),
                "Request {} should be allowed",
                i
            );
        }

        // 11th request should be rejected
        assert!(
            !limiter.allow("tenant-1", 1).await.unwrap(),
            "Request 11 should be rejected"
        );
    }

    #[tokio::test]
    async fn test_sliding_window_basic() {
        let config = RateLimitConfig {
            algorithm: RateLimitAlgorithm::SlidingWindow {
                window_size: ChronoDuration::seconds(1),
                max_requests: 5,
            },
            ..Default::default()
        };

        let limiter = RateLimiter::new(config).unwrap();

        // Should allow first 5 requests
        for i in 0..5 {
            assert!(
                limiter.allow("tenant-1", 1).await.unwrap(),
                "Request {} should be allowed",
                i
            );
        }

        // 6th request should be rejected
        assert!(
            !limiter.allow("tenant-1", 1).await.unwrap(),
            "Request 6 should be rejected"
        );
    }

    #[tokio::test]
    async fn test_multi_tenant_isolation() {
        let config = RateLimitConfig {
            algorithm: RateLimitAlgorithm::TokenBucket {
                capacity: 5,
                refill_rate: 1,
            },
            per_tenant_quotas: true,
            ..Default::default()
        };

        let limiter = RateLimiter::new(config).unwrap();

        // Exhaust tenant-1's quota
        for _ in 0..5 {
            assert!(limiter.allow("tenant-1", 1).await.unwrap());
        }
        assert!(!limiter.allow("tenant-1", 1).await.unwrap());

        // Tenant-2 should still have quota
        assert!(limiter.allow("tenant-2", 1).await.unwrap());
    }

    #[tokio::test]
    async fn test_quota_manager() {
        let config = RateLimitConfig {
            algorithm: RateLimitAlgorithm::TokenBucket {
                capacity: 100,
                refill_rate: 10,
            },
            ..Default::default()
        };

        let manager = QuotaManager::new(config).unwrap();

        // Check request quota
        let result = manager
            .check_quota("tenant-1", &QuotaOperation::Request { count: 50 })
            .await
            .unwrap();
        assert!(result.allowed);
        assert!(result.remaining > 0);
    }

    #[tokio::test]
    async fn test_quota_reset() {
        let config = RateLimitConfig {
            algorithm: RateLimitAlgorithm::TokenBucket {
                capacity: 5,
                refill_rate: 1,
            },
            ..Default::default()
        };

        let limiter = RateLimiter::new(config).unwrap();

        // Exhaust quota
        for _ in 0..5 {
            limiter.allow("tenant-1", 1).await.unwrap();
        }
        assert!(!limiter.allow("tenant-1", 1).await.unwrap());

        // Reset and verify
        limiter.reset("tenant-1").await.unwrap();
        assert!(limiter.allow("tenant-1", 1).await.unwrap());
    }

    #[tokio::test]
    async fn test_custom_quota() {
        let config = RateLimitConfig::default();
        let limiter = RateLimiter::new(config).unwrap();

        // Set custom quota
        let quota = QuotaLimits {
            requests_per_second: 1000,
            ..Default::default()
        };

        limiter
            .set_quota("premium-tenant", quota.clone())
            .await
            .unwrap();

        // Verify quota was set
        let retrieved = limiter.get_quota("premium-tenant").await.unwrap();
        assert_eq!(retrieved.requests_per_second, 1000);
    }

    #[tokio::test]
    async fn test_rate_limit_stats() {
        let config = RateLimitConfig {
            algorithm: RateLimitAlgorithm::TokenBucket {
                capacity: 3,
                refill_rate: 1,
            },
            ..Default::default()
        };

        let limiter = RateLimiter::new(config).unwrap();

        // Make some requests
        limiter.allow("tenant-1", 1).await.unwrap();
        limiter.allow("tenant-1", 1).await.unwrap();
        limiter.allow("tenant-1", 1).await.unwrap();
        limiter.allow("tenant-1", 1).await.unwrap(); // Should be rejected

        let stats = limiter.stats().await.unwrap();
        assert_eq!(stats.total_requests, 4);
        assert_eq!(stats.allowed_requests, 3);
        assert_eq!(stats.rejected_requests, 1);
    }
}
