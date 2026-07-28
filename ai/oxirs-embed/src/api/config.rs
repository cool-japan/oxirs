//! API configuration and state management
//!
//! This module contains configuration structures and server state management
//! for the embedding API service.

#[cfg(feature = "api-server")]
use crate::{CacheManager, EmbeddingModel, ModelRegistry};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Live, aggregate request metrics for the embedding API service.
///
/// These counters are updated by request handlers (via [`ApiMetrics::record`])
/// and read back by the health/info endpoints. They replace the previously
/// hard-coded placeholder values so that reported latency/error figures reflect
/// real observed traffic. Before any request is recorded the derived rates are
/// `0.0`, which is their true value rather than a fabricated constant.
#[derive(Debug, Default)]
pub struct ApiMetrics {
    total_requests: AtomicU64,
    total_errors: AtomicU64,
    total_latency_us: AtomicU64,
}

impl ApiMetrics {
    /// Create an empty metrics tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a completed request with its wall-clock latency and outcome.
    pub fn record(&self, latency: Duration, is_error: bool) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        if is_error {
            self.total_errors.fetch_add(1, Ordering::Relaxed);
        }
        // Saturate to u64 microseconds to avoid overflow on pathological inputs.
        let micros = latency.as_micros().min(u128::from(u64::MAX)) as u64;
        self.total_latency_us.fetch_add(micros, Ordering::Relaxed);
    }

    /// Total number of recorded requests.
    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    /// Mean response time in milliseconds over all recorded requests (0.0 if none).
    pub fn avg_response_time_ms(&self) -> f64 {
        let requests = self.total_requests.load(Ordering::Relaxed);
        if requests == 0 {
            return 0.0;
        }
        let total_us = self.total_latency_us.load(Ordering::Relaxed) as f64;
        (total_us / requests as f64) / 1000.0
    }

    /// Error rate as a percentage over all recorded requests (0.0 if none).
    pub fn error_rate_percent(&self) -> f64 {
        let requests = self.total_requests.load(Ordering::Relaxed);
        if requests == 0 {
            return 0.0;
        }
        let errors = self.total_errors.load(Ordering::Relaxed) as f64;
        (errors / requests as f64) * 100.0
    }
}

/// API server state
#[derive(Clone)]
pub struct ApiState {
    /// Model registry for managing deployed models
    pub registry: Arc<ModelRegistry>,
    /// Cache manager for performance optimization
    pub cache_manager: Arc<CacheManager>,
    /// Currently loaded models
    pub models: Arc<RwLock<HashMap<Uuid, Arc<dyn EmbeddingModel + Send + Sync>>>>,
    /// Live request metrics (latency/error tracking)
    pub metrics: Arc<ApiMetrics>,
    /// API configuration
    pub config: ApiConfig,
}

/// API configuration
#[derive(Debug, Clone)]
pub struct ApiConfig {
    /// Server host
    pub host: String,
    /// Server port
    pub port: u16,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Request timeout in seconds (alias for axum)
    pub request_timeout_secs: u64,
    /// Maximum batch size for bulk operations
    pub max_batch_size: usize,
    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,
    /// Authentication configuration
    pub auth: AuthConfig,
    /// Enable request logging
    pub enable_logging: bool,
    /// Enable CORS
    pub enable_cors: bool,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            timeout_seconds: 30,
            request_timeout_secs: 30,
            max_batch_size: 1000,
            rate_limit: RateLimitConfig::default(),
            auth: AuthConfig::default(),
            enable_logging: true,
            enable_cors: true,
        }
    }
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Requests per minute per IP
    pub requests_per_minute: u32,
    /// Enable rate limiting
    pub enabled: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 1000,
            enabled: true,
        }
    }
}

/// Authentication configuration
#[derive(Debug, Clone, Default)]
pub struct AuthConfig {
    /// Enable API key authentication
    pub require_api_key: bool,
    /// Valid API keys
    pub api_keys: Vec<String>,
    /// Enable JWT authentication
    pub enable_jwt: bool,
    /// JWT secret
    pub jwt_secret: Option<String>,
}
