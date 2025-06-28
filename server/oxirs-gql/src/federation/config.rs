//! Federation configuration types and structures

use serde::{Deserialize, Serialize};

/// Remote GraphQL service endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteEndpoint {
    /// Service identifier
    pub id: String,
    /// GraphQL endpoint URL
    pub url: String,
    /// Optional authentication header
    pub auth_header: Option<String>,
    /// Service namespace for type prefixing
    pub namespace: Option<String>,
    /// Timeout in seconds
    pub timeout_secs: u64,
    /// Maximum retry attempts for failed requests
    pub max_retries: u32,
    /// Retry backoff strategy
    pub retry_strategy: RetryStrategy,
    /// Health check endpoint (optional)
    pub health_check_url: Option<String>,
    /// Service priority (higher priority services are preferred)
    pub priority: i32,
    /// Schema version for backward compatibility tracking
    pub schema_version: Option<String>,
    /// Minimum compatible version with this service
    pub min_compatible_version: Option<String>,
}

/// Retry strategy for failed requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetryStrategy {
    /// No retries
    None,
    /// Fixed delay between retries
    FixedDelay { delay_ms: u64 },
    /// Exponential backoff with jitter
    ExponentialBackoff {
        initial_delay_ms: u64,
        max_delay_ms: u64,
        multiplier: f64,
    },
}

/// Federation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationConfig {
    /// Remote endpoints to federate
    pub endpoints: Vec<RemoteEndpoint>,
    /// Enable schema caching
    pub enable_schema_cache: bool,
    /// Schema cache TTL in seconds
    pub schema_cache_ttl: u64,
    /// Enable query result caching
    pub enable_result_cache: bool,
    /// Query result cache TTL in seconds
    pub result_cache_ttl: u64,
    /// Maximum federation depth for nested queries
    pub max_federation_depth: usize,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            endpoints: Vec::new(),
            enable_schema_cache: true,
            schema_cache_ttl: 3600, // 1 hour
            enable_result_cache: true,
            result_cache_ttl: 300, // 5 minutes
            max_federation_depth: 3,
        }
    }
}
