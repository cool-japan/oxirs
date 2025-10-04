//! Rate limiting for API endpoints

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub requests_per_minute: usize,
    pub tokens_per_minute: usize,
    pub enable_burst: bool,
    pub burst_size: usize,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 60,
            tokens_per_minute: 100_000,
            enable_burst: true,
            burst_size: 10,
        }
    }
}

/// User rate limit state
#[derive(Debug, Clone)]
struct UserRateLimitState {
    request_count: usize,
    token_count: usize,
    window_start: Instant,
    burst_tokens: usize,
}

/// Rate limiter
pub struct RateLimiter {
    config: RateLimitConfig,
    user_states: Arc<RwLock<HashMap<String, UserRateLimitState>>>,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            user_states: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Check if request is allowed
    pub fn check_limit(&self, user_id: &str, tokens: usize) -> Result<bool> {
        let mut user_states = self.user_states.write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;

        let now = Instant::now();
        let window_duration = Duration::from_secs(60);

        let state = user_states
            .entry(user_id.to_string())
            .or_insert_with(|| UserRateLimitState {
                request_count: 0,
                token_count: 0,
                window_start: now,
                burst_tokens: self.config.burst_size,
            });

        // Check if window has expired
        if now.duration_since(state.window_start) >= window_duration {
            // Reset window
            state.request_count = 0;
            state.token_count = 0;
            state.window_start = now;
            state.burst_tokens = self.config.burst_size;
        }

        // Check request limit
        if state.request_count >= self.config.requests_per_minute {
            // Try burst if enabled
            if self.config.enable_burst && state.burst_tokens > 0 {
                state.burst_tokens -= 1;
            } else {
                return Ok(false);
            }
        }

        // Check token limit
        if state.token_count + tokens > self.config.tokens_per_minute {
            return Ok(false);
        }

        // Update counts
        state.request_count += 1;
        state.token_count += tokens;

        Ok(true)
    }

    /// Get remaining quota for user
    pub fn get_remaining_quota(&self, user_id: &str) -> Result<RateLimitQuota> {
        let user_states = self.user_states.read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;

        let state = user_states.get(user_id);

        match state {
            Some(state) => {
                let now = Instant::now();
                let elapsed = now.duration_since(state.window_start);
                let remaining_time = Duration::from_secs(60).saturating_sub(elapsed);

                Ok(RateLimitQuota {
                    remaining_requests: self.config.requests_per_minute.saturating_sub(state.request_count),
                    remaining_tokens: self.config.tokens_per_minute.saturating_sub(state.token_count),
                    reset_in: remaining_time,
                    burst_tokens_remaining: state.burst_tokens,
                })
            }
            None => Ok(RateLimitQuota {
                remaining_requests: self.config.requests_per_minute,
                remaining_tokens: self.config.tokens_per_minute,
                reset_in: Duration::from_secs(60),
                burst_tokens_remaining: self.config.burst_size,
            }),
        }
    }

    /// Reset limit for user (admin function)
    pub fn reset_limit(&self, user_id: &str) -> Result<()> {
        let mut user_states = self.user_states.write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;

        user_states.remove(user_id);
        Ok(())
    }

    /// Clean up expired states
    pub fn cleanup_expired(&self) -> Result<usize> {
        let mut user_states = self.user_states.write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;

        let now = Instant::now();
        let window_duration = Duration::from_secs(60);

        let initial_count = user_states.len();

        user_states.retain(|_, state| {
            now.duration_since(state.window_start) < window_duration * 2
        });

        Ok(initial_count - user_states.len())
    }
}

/// Rate limit quota information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitQuota {
    pub remaining_requests: usize,
    pub remaining_tokens: usize,
    pub reset_in: Duration,
    pub burst_tokens_remaining: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_rate_limiter_allows_within_limit() {
        let config = RateLimitConfig {
            requests_per_minute: 10,
            tokens_per_minute: 1000,
            enable_burst: false,
            burst_size: 0,
        };
        let limiter = RateLimiter::new(config);

        // First request should be allowed
        assert!(limiter.check_limit("user1", 100).unwrap());
    }

    #[test]
    fn test_rate_limiter_blocks_over_limit() {
        let config = RateLimitConfig {
            requests_per_minute: 2,
            tokens_per_minute: 1000,
            enable_burst: false,
            burst_size: 0,
        };
        let limiter = RateLimiter::new(config);

        // First two requests allowed
        assert!(limiter.check_limit("user1", 100).unwrap());
        assert!(limiter.check_limit("user1", 100).unwrap());

        // Third should be blocked
        assert!(!limiter.check_limit("user1", 100).unwrap());
    }

    #[test]
    fn test_burst_mode() {
        let config = RateLimitConfig {
            requests_per_minute: 2,
            tokens_per_minute: 1000,
            enable_burst: true,
            burst_size: 1,
        };
        let limiter = RateLimiter::new(config);

        // First two requests allowed
        assert!(limiter.check_limit("user1", 100).unwrap());
        assert!(limiter.check_limit("user1", 100).unwrap());

        // Third allowed due to burst
        assert!(limiter.check_limit("user1", 100).unwrap());

        // Fourth blocked (burst exhausted)
        assert!(!limiter.check_limit("user1", 100).unwrap());
    }

    #[test]
    fn test_token_limit() {
        let config = RateLimitConfig {
            requests_per_minute: 100,
            tokens_per_minute: 1000,
            enable_burst: false,
            burst_size: 0,
        };
        let limiter = RateLimiter::new(config);

        // Request with 600 tokens
        assert!(limiter.check_limit("user1", 600).unwrap());

        // Request with 500 more tokens should be blocked (would exceed 1000)
        assert!(!limiter.check_limit("user1", 500).unwrap());
    }

    #[test]
    fn test_get_remaining_quota() {
        let config = RateLimitConfig {
            requests_per_minute: 10,
            tokens_per_minute: 1000,
            enable_burst: false,
            burst_size: 0,
        };
        let limiter = RateLimiter::new(config);

        // Make one request
        limiter.check_limit("user1", 100).unwrap();

        let quota = limiter.get_remaining_quota("user1").unwrap();
        assert_eq!(quota.remaining_requests, 9);
        assert_eq!(quota.remaining_tokens, 900);
    }

    #[test]
    fn test_separate_user_limits() {
        let config = RateLimitConfig::default();
        let limiter = RateLimiter::new(config);

        assert!(limiter.check_limit("user1", 100).unwrap());
        assert!(limiter.check_limit("user2", 100).unwrap());

        // Each user has their own limit
        let quota1 = limiter.get_remaining_quota("user1").unwrap();
        let quota2 = limiter.get_remaining_quota("user2").unwrap();

        assert_eq!(quota1.remaining_requests, quota2.remaining_requests);
    }
}
