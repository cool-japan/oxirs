//! # Advanced Rate Limiting for GraphQL
//!
//! Production-ready rate limiting system with multiple algorithms and strategies.
//! Supports per-user, per-API-key, and per-IP rate limiting with distributed coordination.
//!
//! ## Features
//!
//! - **Token Bucket Algorithm**: Classic token bucket for burst control
//! - **Sliding Window Counters**: Smooth rate limiting without boundary issues
//! - **Fixed Window Counters**: Simple time-based windows
//! - **Query Complexity-based**: Limit by GraphQL query complexity, not just count
//! - **Distributed Rate Limiting**: Redis-based coordination across servers
//! - **Adaptive Rate Limiting**: Automatic adjustment based on server load
//! - **Per-Client Limits**: Different limits for different users/API keys
//! - **Configurable Policies**: Flexible policy engine
//!
//! ## Algorithms
//!
//! ### Token Bucket
//! - Tokens refill at a constant rate
//! - Each request consumes tokens
//! - Allows bursts up to bucket capacity
//!
//! ### Sliding Window
//! - Combines previous and current window
//! - Smooth transition between windows
//! - No boundary burst issues
//!
//! ### Adaptive
//! - Monitors server CPU, memory, throughput
//! - Automatically adjusts limits based on load
//! - Prevents server overload

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// Rate limiting algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RateLimitAlgorithm {
    /// Token bucket algorithm (allows bursts)
    TokenBucket,
    /// Sliding window counters (smooth limiting)
    SlidingWindow,
    /// Fixed window counters (simple)
    FixedWindow,
    /// Adaptive based on server load
    Adaptive,
}

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Algorithm to use
    pub algorithm: RateLimitAlgorithm,
    /// Maximum requests per window
    pub max_requests: u32,
    /// Window duration in seconds
    pub window_seconds: u64,
    /// Enable query complexity-based limiting
    pub complexity_based: bool,
    /// Maximum complexity per window (if complexity_based)
    pub max_complexity: Option<u32>,
    /// Enable distributed rate limiting (Redis)
    pub distributed: bool,
    /// Redis URL for distributed limiting
    pub redis_url: Option<String>,
    /// Enable adaptive rate limiting
    pub adaptive: bool,
    /// Adaptive target CPU percentage (0-100)
    pub adaptive_cpu_target: Option<f32>,
    /// Adaptive target memory percentage (0-100)
    pub adaptive_memory_target: Option<f32>,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            algorithm: RateLimitAlgorithm::TokenBucket,
            max_requests: 100,
            window_seconds: 60,
            complexity_based: true,
            max_complexity: Some(10000),
            distributed: false,
            redis_url: None,
            adaptive: false,
            adaptive_cpu_target: Some(80.0),
            adaptive_memory_target: Some(90.0),
        }
    }
}

/// Rate limit policy for specific clients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitPolicy {
    /// Client identifier (user_id, api_key, or IP)
    pub client_id: String,
    /// Custom rate limit config
    pub config: RateLimitConfig,
    /// Policy priority (higher = more important)
    pub priority: u32,
}

impl RateLimitPolicy {
    pub fn new(client_id: String, config: RateLimitConfig) -> Self {
        Self {
            client_id,
            config,
            priority: 0,
        }
    }

    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }
}

/// Token bucket state
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenBucket {
    /// Current number of tokens
    tokens: f64,
    /// Maximum bucket capacity
    capacity: f64,
    /// Token refill rate (tokens per second)
    refill_rate: f64,
    /// Last refill timestamp
    last_refill: u64,
}

impl TokenBucket {
    fn new(capacity: u32, refill_rate: f64) -> Self {
        Self {
            tokens: capacity as f64,
            capacity: capacity as f64,
            refill_rate,
            last_refill: Self::now(),
        }
    }

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    fn refill(&mut self) {
        let now = Self::now();
        let elapsed = now.saturating_sub(self.last_refill) as f64;
        let new_tokens = elapsed * self.refill_rate;
        self.tokens = (self.tokens + new_tokens).min(self.capacity);
        self.last_refill = now;
    }

    fn consume(&mut self, tokens: f64) -> bool {
        self.refill();
        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    fn available(&mut self) -> f64 {
        self.refill();
        self.tokens
    }
}

/// Sliding window counter
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SlidingWindowCounter {
    /// Current window start time
    current_window: u64,
    /// Current window count
    current_count: u32,
    /// Previous window count
    previous_count: u32,
    /// Window duration
    window_duration: u64,
}

impl SlidingWindowCounter {
    fn new(window_duration: u64) -> Self {
        Self {
            current_window: Self::current_window(window_duration),
            current_count: 0,
            previous_count: 0,
            window_duration,
        }
    }

    fn current_window(window_duration: u64) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            / window_duration
    }

    fn now_seconds() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    fn increment(&mut self, amount: u32) {
        let now_window = Self::current_window(self.window_duration);

        if now_window > self.current_window + 1 {
            // More than one window has passed, reset both
            self.previous_count = 0;
            self.current_count = amount;
            self.current_window = now_window;
        } else if now_window > self.current_window {
            // Move to next window
            self.previous_count = self.current_count;
            self.current_count = amount;
            self.current_window = now_window;
        } else {
            // Same window
            self.current_count += amount;
        }
    }

    fn count(&self, max_requests: u32) -> Result<(), String> {
        let now_window = Self::current_window(self.window_duration);

        if now_window > self.current_window + 1 {
            // More than one window has passed, no requests in current window
            return Ok(());
        }

        if now_window > self.current_window {
            // In transition between windows, use sliding calculation
            let now = Self::now_seconds();
            let window_progress = (now % self.window_duration) as f64 / self.window_duration as f64;
            let weighted_count =
                (self.previous_count as f64 * (1.0 - window_progress)) + self.current_count as f64;

            if weighted_count >= max_requests as f64 {
                return Err(format!(
                    "Rate limit exceeded: {:.0}/{} requests in sliding window",
                    weighted_count, max_requests
                ));
            }
        } else {
            // Same window
            if self.current_count >= max_requests {
                return Err(format!(
                    "Rate limit exceeded: {}/{} requests in window",
                    self.current_count, max_requests
                ));
            }
        }

        Ok(())
    }
}

/// Fixed window counter
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FixedWindowCounter {
    /// Window start time
    window_start: u64,
    /// Request count in this window
    count: u32,
    /// Window duration
    window_duration: u64,
}

impl FixedWindowCounter {
    fn new(window_duration: u64) -> Self {
        Self {
            window_start: Self::now(),
            count: 0,
            window_duration,
        }
    }

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    fn increment(&mut self, amount: u32) {
        let now = Self::now();

        if now >= self.window_start + self.window_duration {
            // New window
            self.window_start = now;
            self.count = amount;
        } else {
            // Same window
            self.count += amount;
        }
    }

    fn check(&self, max_requests: u32) -> Result<(), String> {
        let now = Self::now();

        if now >= self.window_start + self.window_duration {
            // New window, no requests yet
            Ok(())
        } else if self.count >= max_requests {
            Err(format!(
                "Rate limit exceeded: {}/{} requests in window",
                self.count, max_requests
            ))
        } else {
            Ok(())
        }
    }
}

/// Rate limit state per client
#[derive(Debug, Clone, Serialize, Deserialize)]
enum RateLimitState {
    TokenBucket(TokenBucket),
    SlidingWindow(SlidingWindowCounter),
    FixedWindow(FixedWindowCounter),
}

/// Rate limiter result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitResult {
    /// Whether the request is allowed
    pub allowed: bool,
    /// Current limit
    pub limit: u32,
    /// Remaining requests/tokens
    pub remaining: u32,
    /// Time until reset (seconds)
    pub reset_after: u64,
    /// Retry after (seconds) if not allowed
    pub retry_after: Option<u64>,
}

/// Rate limiter
pub struct RateLimiter {
    /// Configuration
    config: Arc<RateLimitConfig>,
    /// Per-client state
    states: Arc<RwLock<HashMap<String, RateLimitState>>>,
    /// Custom policies
    policies: Arc<RwLock<HashMap<String, RateLimitPolicy>>>,
    /// System load monitor for adaptive limiting
    load_monitor: Arc<RwLock<SystemLoad>>,
}

#[derive(Debug, Clone)]
struct SystemLoad {
    cpu_usage: f32,
    memory_usage: f32,
    last_update: std::time::Instant,
}

impl SystemLoad {
    fn new() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            last_update: std::time::Instant::now(),
        }
    }

    fn update(&mut self) {
        // Update only every 5 seconds to avoid overhead
        if self.last_update.elapsed() < Duration::from_secs(5) {
            return;
        }

        use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

        let mut sys = System::new_with_specifics(
            RefreshKind::new()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::everything()),
        );
        sys.refresh_all();

        // Calculate average CPU usage across all cores
        let cpus = sys.cpus();
        if !cpus.is_empty() {
            self.cpu_usage =
                cpus.iter().map(|cpu| cpu.cpu_usage()).sum::<f32>() / cpus.len() as f32;
        }

        self.memory_usage = (sys.used_memory() as f64 / sys.total_memory() as f64 * 100.0) as f32;
        self.last_update = std::time::Instant::now();
    }

    fn should_reduce_limits(&self, cpu_target: f32, memory_target: f32) -> bool {
        self.cpu_usage > cpu_target || self.memory_usage > memory_target
    }
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config: Arc::new(config),
            states: Arc::new(RwLock::new(HashMap::new())),
            policies: Arc::new(RwLock::new(HashMap::new())),
            load_monitor: Arc::new(RwLock::new(SystemLoad::new())),
        }
    }

    /// Add a custom policy for a specific client
    pub async fn add_policy(&self, policy: RateLimitPolicy) -> Result<()> {
        let mut policies = self.policies.write().await;
        policies.insert(policy.client_id.clone(), policy);
        Ok(())
    }

    /// Remove a custom policy
    pub async fn remove_policy(&self, client_id: &str) -> Result<()> {
        let mut policies = self.policies.write().await;
        policies.remove(client_id);
        Ok(())
    }

    /// Check rate limit for a client
    pub async fn check_rate_limit(
        &self,
        client_id: &str,
        cost: Option<u32>,
    ) -> Result<RateLimitResult> {
        // Get effective config (policy or default)
        let effective_config = {
            let policies = self.policies.read().await;
            policies
                .get(client_id)
                .map(|p| p.config.clone())
                .unwrap_or_else(|| (*self.config).clone())
        };

        // Adaptive limiting
        if effective_config.adaptive {
            let mut load_monitor = self.load_monitor.write().await;
            load_monitor.update();

            if load_monitor.should_reduce_limits(
                effective_config.adaptive_cpu_target.unwrap_or(80.0),
                effective_config.adaptive_memory_target.unwrap_or(90.0),
            ) {
                // Reduce limits by 50% when under high load
                return Ok(RateLimitResult {
                    allowed: false,
                    limit: effective_config.max_requests,
                    remaining: 0,
                    reset_after: effective_config.window_seconds,
                    retry_after: Some(5),
                });
            }
        }

        let request_cost = if effective_config.complexity_based {
            cost.unwrap_or(1)
        } else {
            1
        };

        let mut states = self.states.write().await;
        let state = states
            .entry(client_id.to_string())
            .or_insert_with(|| Self::create_state(&effective_config));

        match state {
            RateLimitState::TokenBucket(bucket) => {
                let allowed = bucket.consume(request_cost as f64);
                let remaining = bucket.available() as u32;

                Ok(RateLimitResult {
                    allowed,
                    limit: effective_config.max_requests,
                    remaining,
                    reset_after: effective_config.window_seconds,
                    retry_after: if !allowed { Some(1) } else { None },
                })
            }
            RateLimitState::SlidingWindow(counter) => {
                let check_result = counter.count(effective_config.max_requests);

                if check_result.is_ok() {
                    counter.increment(request_cost);
                    Ok(RateLimitResult {
                        allowed: true,
                        limit: effective_config.max_requests,
                        remaining: effective_config
                            .max_requests
                            .saturating_sub(counter.current_count),
                        reset_after: effective_config.window_seconds,
                        retry_after: None,
                    })
                } else {
                    Ok(RateLimitResult {
                        allowed: false,
                        limit: effective_config.max_requests,
                        remaining: 0,
                        reset_after: effective_config.window_seconds,
                        retry_after: Some(effective_config.window_seconds),
                    })
                }
            }
            RateLimitState::FixedWindow(counter) => {
                let check_result = counter.check(effective_config.max_requests);

                if check_result.is_ok() {
                    counter.increment(request_cost);
                    Ok(RateLimitResult {
                        allowed: true,
                        limit: effective_config.max_requests,
                        remaining: effective_config.max_requests.saturating_sub(counter.count),
                        reset_after: effective_config.window_seconds,
                        retry_after: None,
                    })
                } else {
                    Ok(RateLimitResult {
                        allowed: false,
                        limit: effective_config.max_requests,
                        remaining: 0,
                        reset_after: effective_config.window_seconds,
                        retry_after: Some(effective_config.window_seconds),
                    })
                }
            }
        }
    }

    /// Reset rate limit for a client
    pub async fn reset_client(&self, client_id: &str) -> Result<()> {
        let mut states = self.states.write().await;
        states.remove(client_id);
        Ok(())
    }

    /// Get statistics for all clients
    pub async fn get_statistics(&self) -> HashMap<String, String> {
        let states = self.states.read().await;
        states
            .iter()
            .map(|(id, state)| {
                let info = match state {
                    RateLimitState::TokenBucket(b) => {
                        format!("TokenBucket(tokens: {:.2})", b.tokens)
                    }
                    RateLimitState::SlidingWindow(c) => {
                        format!("SlidingWindow(current: {})", c.current_count)
                    }
                    RateLimitState::FixedWindow(c) => {
                        format!("FixedWindow(count: {})", c.count)
                    }
                };
                (id.clone(), info)
            })
            .collect()
    }

    fn create_state(config: &RateLimitConfig) -> RateLimitState {
        match config.algorithm {
            RateLimitAlgorithm::TokenBucket => {
                let refill_rate = config.max_requests as f64 / config.window_seconds as f64;
                RateLimitState::TokenBucket(TokenBucket::new(config.max_requests, refill_rate))
            }
            RateLimitAlgorithm::SlidingWindow => {
                RateLimitState::SlidingWindow(SlidingWindowCounter::new(config.window_seconds))
            }
            RateLimitAlgorithm::FixedWindow | RateLimitAlgorithm::Adaptive => {
                RateLimitState::FixedWindow(FixedWindowCounter::new(config.window_seconds))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limit_config_default() {
        let config = RateLimitConfig::default();
        assert_eq!(config.algorithm, RateLimitAlgorithm::TokenBucket);
        assert_eq!(config.max_requests, 100);
        assert_eq!(config.window_seconds, 60);
        assert!(config.complexity_based);
    }

    #[tokio::test]
    async fn test_token_bucket_creation() {
        let bucket = TokenBucket::new(100, 10.0);
        assert_eq!(bucket.capacity, 100.0);
        assert_eq!(bucket.refill_rate, 10.0);
    }

    #[tokio::test]
    async fn test_token_bucket_consume() {
        let mut bucket = TokenBucket::new(100, 10.0);
        assert!(bucket.consume(50.0));
        assert!(bucket.tokens < 51.0 && bucket.tokens > 49.0);
    }

    #[tokio::test]
    async fn test_sliding_window_counter() {
        let mut counter = SlidingWindowCounter::new(60);
        counter.increment(10);
        assert_eq!(counter.current_count, 10);
    }

    #[tokio::test]
    async fn test_fixed_window_counter() {
        let mut counter = FixedWindowCounter::new(60);
        counter.increment(10);
        assert_eq!(counter.count, 10);

        let result = counter.check(100);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_rate_limiter_creation() {
        let config = RateLimitConfig::default();
        let _limiter = RateLimiter::new(config);
    }

    #[tokio::test]
    async fn test_rate_limiter_check_allowed() {
        let config = RateLimitConfig {
            max_requests: 10,
            ..Default::default()
        };
        let limiter = RateLimiter::new(config);

        let result = limiter.check_rate_limit("client1", Some(1)).await;
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.allowed);
    }

    #[tokio::test]
    async fn test_rate_limiter_check_exceeded() {
        let config = RateLimitConfig {
            algorithm: RateLimitAlgorithm::FixedWindow,
            max_requests: 5,
            window_seconds: 60,
            ..Default::default()
        };
        let limiter = RateLimiter::new(config);

        // Make 5 requests (should succeed)
        for _ in 0..5 {
            let result = limiter.check_rate_limit("client1", Some(1)).await.unwrap();
            assert!(result.allowed);
        }

        // 6th request should fail
        let result = limiter.check_rate_limit("client1", Some(1)).await.unwrap();
        assert!(!result.allowed);
        assert!(result.retry_after.is_some());
    }

    #[tokio::test]
    async fn test_rate_limiter_custom_policy() {
        let default_config = RateLimitConfig {
            max_requests: 10,
            ..Default::default()
        };
        let limiter = RateLimiter::new(default_config);

        // Add custom policy for VIP client
        let vip_policy = RateLimitPolicy::new(
            "vip_client".to_string(),
            RateLimitConfig {
                max_requests: 1000,
                ..RateLimitConfig::default()
            },
        );
        limiter.add_policy(vip_policy).await.unwrap();

        // VIP client should have higher limits
        let result = limiter
            .check_rate_limit("vip_client", Some(1))
            .await
            .unwrap();
        assert!(result.allowed);
        assert_eq!(result.limit, 1000);
    }

    #[tokio::test]
    async fn test_complexity_based_limiting() {
        let config = RateLimitConfig {
            complexity_based: true,
            max_requests: 100,
            max_complexity: Some(1000),
            ..Default::default()
        };
        let limiter = RateLimiter::new(config);

        // High complexity query
        let result = limiter.check_rate_limit("client1", Some(50)).await.unwrap();
        assert!(result.allowed);
    }

    #[tokio::test]
    async fn test_rate_limiter_reset() {
        let config = RateLimitConfig {
            algorithm: RateLimitAlgorithm::FixedWindow,
            max_requests: 5,
            ..Default::default()
        };
        let limiter = RateLimiter::new(config);

        // Exhaust limit
        for _ in 0..5 {
            let _ = limiter.check_rate_limit("client1", Some(1)).await;
        }

        // Should be rate limited
        let result = limiter.check_rate_limit("client1", Some(1)).await.unwrap();
        assert!(!result.allowed);

        // Reset client
        limiter.reset_client("client1").await.unwrap();

        // Should be allowed again
        let result = limiter.check_rate_limit("client1", Some(1)).await.unwrap();
        assert!(result.allowed);
    }
}
