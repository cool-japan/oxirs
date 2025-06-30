//! API configuration and state management
//!
//! This module contains configuration structures and server state management
//! for the embedding API service.

#[cfg(feature = "api-server")]
use crate::{CacheManager, EmbeddingModel, ModelRegistry};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// API server state
#[derive(Clone)]
pub struct ApiState {
    /// Model registry for managing deployed models
    pub registry: Arc<ModelRegistry>,
    /// Cache manager for performance optimization
    pub cache_manager: Arc<CacheManager>,
    /// Currently loaded models
    pub models: Arc<RwLock<HashMap<Uuid, Arc<dyn EmbeddingModel + Send + Sync>>>>,
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
#[derive(Debug, Clone)]
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

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            require_api_key: false,
            api_keys: Vec::new(),
            enable_jwt: false,
            jwt_secret: None,
        }
    }
}