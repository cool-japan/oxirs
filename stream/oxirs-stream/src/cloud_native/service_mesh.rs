//! Service Mesh Integration Types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
pub struct ServiceMeshConfig {
    /// Service mesh provider
    pub provider: ServiceMeshProvider,
    /// Enable service mesh
    pub enabled: bool,
    /// mTLS configuration
    pub mtls: MutualTLSConfig,
    /// Traffic management
    pub traffic_management: TrafficManagementConfig,
    /// Observability configuration
    pub observability: ServiceMeshObservabilityConfig,
    /// Security policies
    pub security_policies: SecurityPolicyConfig,
}

impl Default for ServiceMeshConfig {
    fn default() -> Self {
        Self {
            provider: ServiceMeshProvider::Istio,
            enabled: true,
            mtls: MutualTLSConfig::default(),
            traffic_management: TrafficManagementConfig::default(),
            observability: ServiceMeshObservabilityConfig::default(),
            security_policies: SecurityPolicyConfig::default(),
        }
    }
}

/// Service mesh providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceMeshProvider {
    Istio,
    Linkerd,
    ConsulConnect,
    OpenServiceMesh,
    Kuma,
}

/// Mutual TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutualTLSConfig {
    /// Enable mTLS
    pub enabled: bool,
    /// mTLS mode
    pub mode: MutualTLSMode,
    /// Certificate authority
    pub ca_provider: CertificateAuthorityProvider,
    /// Certificate rotation interval
    pub cert_rotation_interval: ChronoDuration,
    /// Key size
    pub key_size: u32,
}

impl Default for MutualTLSConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: MutualTLSMode::Strict,
            ca_provider: CertificateAuthorityProvider::Istio,
            cert_rotation_interval: ChronoDuration::days(30),
            key_size: 2048,
        }
    }
}

/// Mutual TLS modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutualTLSMode {
    Disabled,
    Permissive,
    Strict,
}

/// Certificate authority providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CertificateAuthorityProvider {
    Istio,
    CertManager,
    Vault,
    External,
}

/// Traffic management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficManagementConfig {
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
    /// Retry configuration
    pub retry: RetryConfig,
    /// Timeout configuration
    pub timeout: TimeoutConfig,
    /// Rate limiting
    pub rate_limiting: ServiceMeshRateLimitConfig,
}

impl Default for TrafficManagementConfig {
    fn default() -> Self {
        Self {
            load_balancing: LoadBalancingStrategy::RoundRobin,
            circuit_breaker: CircuitBreakerConfig::default(),
            retry: RetryConfig::default(),
            timeout: TimeoutConfig::default(),
            rate_limiting: ServiceMeshRateLimitConfig::default(),
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnection,
    Random,
    WeightedRoundRobin,
    ConsistentHash,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Enable circuit breaker
    pub enabled: bool,
    /// Consecutive errors threshold
    pub consecutive_errors: u32,
    /// Error threshold percentage
    pub error_threshold_percentage: f64,
    /// Minimum request threshold
    pub min_request_amount: u32,
    /// Sleep window
    pub sleep_window: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            consecutive_errors: 5,
            error_threshold_percentage: 50.0,
            min_request_amount: 20,
            sleep_window: Duration::from_secs(30),
        }
    }
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Number of retry attempts
    pub attempts: u32,
    /// Per-try timeout
    pub per_try_timeout: Duration,
    /// Retry conditions
    pub retry_on: Vec<RetryCondition>,
    /// Backoff strategy
    pub backoff: BackoffStrategy,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            attempts: 3,
            per_try_timeout: Duration::from_secs(5),
            retry_on: vec![
                RetryCondition::FiveXX,
                RetryCondition::GatewayError,
                RetryCondition::ConnectFailure,
            ],
            backoff: BackoffStrategy::Exponential {
                base_interval: Duration::from_millis(25),
                max_interval: Duration::from_secs(30),
            },
        }
    }
}

/// Retry conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetryCondition {
    FiveXX,
    GatewayError,
    ConnectFailure,
    RefusedStream,
    Cancelled,
    DeadlineExceeded,
    ResourceExhausted,
}

/// Backoff strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackoffStrategy {
    Fixed {
        interval: Duration,
    },
    Exponential {
        base_interval: Duration,
        max_interval: Duration,
    },
}

/// Timeout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Request timeout
    pub request_timeout: Duration,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Stream idle timeout
    pub stream_idle_timeout: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(30),
            connection_timeout: Duration::from_secs(10),
            stream_idle_timeout: Duration::from_secs(300),
        }
    }
}

/// Service mesh rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMeshRateLimitConfig {
    /// Enable rate limiting
    pub enabled: bool,
    /// Rate limit per second
    pub requests_per_second: u32,
    /// Burst size
    pub burst_size: u32,
    /// Rate limit headers
    pub fill_interval: Duration,
}

impl Default for ServiceMeshRateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            requests_per_second: 100,
            burst_size: 200,
            fill_interval: Duration::from_secs(1),
        }
    }
}

/// Service mesh observability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMeshObservabilityConfig {
    /// Enable distributed tracing
    pub tracing: TracingConfig,
    /// Enable metrics collection
    pub metrics: MetricsConfig,
    /// Enable access logging
    pub access_logs: AccessLogsConfig,
}

impl Default for ServiceMeshObservabilityConfig {
    fn default() -> Self {
        Self {
            tracing: TracingConfig::default(),
            metrics: MetricsConfig::default(),
            access_logs: AccessLogsConfig::default(),
        }
    }
}
