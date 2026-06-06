//! Runtime-domain configuration types: rate limiting, CORS, sessions, metrics, observability,
//! performance, caching, and logging.
use serde::{Deserialize, Serialize};
use validator::Validate;

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct RateLimitConfig {
    #[validate(range(min = 1))]
    pub requests_per_minute: u32,

    #[validate(range(min = 1))]
    pub burst_size: u32,

    pub per_ip: bool,

    pub per_user: bool,

    pub whitelist: Vec<String>,
}

/// CORS configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct CorsConfig {
    pub enabled: bool,
    pub allow_origins: Vec<String>,
    pub allow_methods: Vec<String>,
    pub allow_headers: Vec<String>,
    pub expose_headers: Vec<String>,
    pub allow_credentials: bool,

    #[validate(range(min = 0, max = 86400))]
    pub max_age_secs: u64,
}

/// Session configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct SessionConfig {
    #[validate(length(min = 32))]
    pub secret: String,

    #[validate(range(min = 300, max = 86400))] // 5 min to 24 hours
    pub timeout_secs: u64,

    pub secure: bool,

    pub http_only: bool,

    pub same_site: SameSitePolicy,
}

/// SameSite cookie policy
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SameSitePolicy {
    Strict,
    Lax,
    None,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate, Default)]
pub struct MonitoringConfig {
    pub metrics: MetricsConfig,
    pub health_checks: HealthCheckConfig,
    pub tracing: TracingConfig,
    pub prometheus: Option<PrometheusConfig>,
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct MetricsConfig {
    pub enabled: bool,

    #[validate(length(min = 1))]
    pub endpoint: String,

    #[validate(range(min = 1, max = 65535))]
    pub port: Option<u16>,

    pub namespace: String,

    pub collect_system_metrics: bool,

    pub histogram_buckets: Vec<f64>,
}

/// Prometheus configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct PrometheusConfig {
    pub enabled: bool,

    #[validate(length(min = 1))]
    pub endpoint: String,

    #[validate(range(min = 1, max = 65535))]
    pub port: Option<u16>,

    pub namespace: String,

    pub job_name: String,

    pub instance: String,

    pub scrape_interval_secs: u64,

    pub timeout_secs: u64,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct HealthCheckConfig {
    pub enabled: bool,

    #[validate(range(min = 1))]
    pub interval_secs: u64,

    #[validate(range(min = 1))]
    pub timeout_secs: u64,

    pub checks: Vec<String>,
}

/// Tracing configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct TracingConfig {
    pub enabled: bool,

    pub endpoint: Option<String>,

    pub service_name: String,

    pub sample_rate: f64,

    pub output: TracingOutput,
}

/// Tracing output options
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TracingOutput {
    Stdout,
    Stderr,
    File,
    Jaeger,
    Otlp,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct PerformanceConfig {
    pub caching: CacheConfig,
    pub connection_pool: ConnectionPoolConfig,
    pub query_optimization: QueryOptimizationConfig,
    #[validate(nested)]
    pub rate_limiting: Option<RateLimitConfig>,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct CacheConfig {
    pub enabled: bool,

    #[validate(range(min = 1))]
    pub max_size: usize,

    #[validate(range(min = 1))]
    pub ttl_secs: u64,

    pub query_cache_enabled: bool,

    pub result_cache_enabled: bool,

    pub plan_cache_enabled: bool,
}

/// Connection pool configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ConnectionPoolConfig {
    #[validate(range(min = 1))]
    pub min_connections: usize,

    #[validate(range(min = 1))]
    pub max_connections: usize,

    #[validate(range(min = 1))]
    pub connection_timeout_secs: u64,

    #[validate(range(min = 1))]
    pub idle_timeout_secs: u64,

    #[validate(range(min = 1))]
    pub max_lifetime_secs: u64,
}

/// Query optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct QueryOptimizationConfig {
    pub enabled: bool,

    #[validate(range(min = 1))]
    pub max_query_time_secs: u64,

    #[validate(range(min = 1))]
    pub max_result_size: usize,

    pub parallel_execution: bool,

    #[validate(range(min = 1))]
    pub thread_pool_size: usize,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct LoggingConfig {
    #[validate(length(min = 1))]
    pub level: String,

    pub format: LogFormat,

    pub output: LogOutput,

    pub file_config: Option<FileLogConfig>,
}

/// Log format options
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    Text,
    Json,
    Compact,
}

/// Log output options
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogOutput {
    Stdout,
    Stderr,
    File,
    Both,
}

/// File logging configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct FileLogConfig {
    pub path: std::path::PathBuf,

    #[validate(range(min = 1))]
    pub max_size_mb: u64,

    #[validate(range(min = 1))]
    pub max_files: usize,

    pub compress: bool,
}

// ---- Default implementations ----

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: "/metrics".to_string(),
            port: None,
            namespace: "oxirs".to_string(),
            collect_system_metrics: true,
            histogram_buckets: vec![0.001, 0.01, 0.1, 1.0, 10.0],
        }
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_secs: 30,
            timeout_secs: 5,
            checks: vec!["database".to_string(), "memory".to_string()],
        }
    }
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: None,
            service_name: "oxirs-fuseki".to_string(),
            sample_rate: 1.0,
            output: TracingOutput::Stdout,
        }
    }
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            allow_origins: vec!["*".to_string()],
            allow_methods: vec!["GET".to_string(), "POST".to_string(), "OPTIONS".to_string()],
            allow_headers: vec!["Content-Type".to_string(), "Authorization".to_string()],
            expose_headers: vec![],
            allow_credentials: false,
            max_age_secs: 3600,
        }
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            secret: "default-session-secret-change-in-production".to_string(),
            timeout_secs: 3600,
            secure: false,
            http_only: true,
            same_site: SameSitePolicy::Lax,
        }
    }
}
