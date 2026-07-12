//! Network, server, security, and environment OxiRS configuration types.
//!
//! Contains `ConcurrencyConfig`, `MonitoringConfig`, `SecurityConfig`,
//! `OptimizationConfig`, `Environment`, `ConfigSource`, `ConfigWatcher`,
//! `ConfigError`, and all subordinate types + Default impls.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────
// Environment / ConfigSource / ConfigWatcher / ConfigError
// ─────────────────────────────────────────────────────────────

/// Environment types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Environment {
    Development,
    Testing,
    Staging,
    Production,
}

/// Configuration sources
#[derive(Debug, Clone)]
pub enum ConfigSource {
    File { path: PathBuf },
    Environment,
    CommandLine { args: Vec<String> },
    Remote { url: String },
    Database { connection: String },
}

/// Configuration errors
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Configuration file not found: {0}")]
    FileNotFound(PathBuf),
    #[error("Invalid configuration format: {0}")]
    InvalidFormat(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Environment variable error: {0}")]
    EnvironmentError(String),
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

// ─────────────────────────────────────────────────────────────
// ConcurrencyConfig
// ─────────────────────────────────────────────────────────────

/// Concurrency configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConcurrencyConfig {
    /// Thread pool configuration
    pub thread_pool: ThreadPoolConfig,
    /// Lock configuration
    pub locks: LockConfig,
    /// Async runtime configuration
    pub async_runtime: AsyncRuntimeConfig,
}

/// Thread pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPoolConfig {
    /// Number of worker threads (0 = auto-detect)
    pub worker_threads: usize,
    /// Thread stack size
    pub stack_size: usize,
    /// Thread priority
    pub priority: ThreadPriority,
    /// Enable work stealing
    pub work_stealing: bool,
}

/// Thread priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreadPriority {
    Low,
    Normal,
    High,
    Realtime,
}

/// Lock configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockConfig {
    /// Default lock type
    pub default_type: LockType,
    /// Lock timeout
    pub timeout: Duration,
    /// Enable lock debugging
    pub enable_debugging: bool,
    /// Deadlock detection
    pub deadlock_detection: bool,
}

/// Lock types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LockType {
    Mutex,
    RwLock,
    SpinLock,
    AtomicLock,
}

/// Async runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncRuntimeConfig {
    /// Runtime type
    pub runtime_type: AsyncRuntimeType,
    /// Enable I/O driver
    pub enable_io: bool,
    /// Enable time driver
    pub enable_time: bool,
    /// Worker thread configuration
    pub worker_config: AsyncWorkerConfig,
}

/// Async runtime types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AsyncRuntimeType {
    CurrentThread,
    MultiThread,
    MultiThreadAlt,
}

/// Async worker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncWorkerConfig {
    /// Core worker threads
    pub core_threads: usize,
    /// Maximum worker threads
    pub max_threads: usize,
    /// Thread keep-alive time
    pub keep_alive: Duration,
    /// Thread name prefix
    pub thread_name_prefix: String,
}

// ─────────────────────────────────────────────────────────────
// MonitoringConfig
// ─────────────────────────────────────────────────────────────

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable monitoring
    pub enabled: bool,
    /// Metrics configuration
    pub metrics: MetricsConfig,
    /// Tracing configuration
    pub tracing: TracingConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// Health check configuration
    pub health_checks: HealthCheckConfig,
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Metrics exporter type
    pub exporter: MetricsExporter,
    /// Collection interval
    pub collection_interval: Duration,
    /// Retention period
    pub retention_period: Duration,
    /// Custom metrics
    pub custom_metrics: Vec<CustomMetric>,
}

/// Metrics exporters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricsExporter {
    None,
    Prometheus,
    StatsD,
    OpenMetrics,
    Custom,
}

/// Custom metric definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMetric {
    /// Metric name
    pub name: String,
    /// Metric type
    pub metric_type: MetricType,
    /// Description
    pub description: String,
    /// Labels
    pub labels: Vec<String>,
}

/// Metric types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

/// Tracing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    /// Enable tracing
    pub enabled: bool,
    /// Trace exporter
    pub exporter: TraceExporter,
    /// Sampling rate (0.0 to 1.0)
    pub sampling_rate: f64,
    /// Maximum span count
    pub max_spans: usize,
}

/// Trace exporters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceExporter {
    None,
    Jaeger,
    Zipkin,
    OpenTelemetry,
    Console,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: LogLevel,
    /// Log format
    pub format: LogFormat,
    /// Log targets
    pub targets: Vec<LogTarget>,
    /// Enable structured logging
    pub structured: bool,
}

/// Log levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// Log formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogFormat {
    Plain,
    Json,
    Logfmt,
    Custom,
}

/// Log targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogTarget {
    Console,
    File {
        path: PathBuf,
        rotation: FileRotation,
    },
    Syslog {
        facility: String,
    },
    Network {
        endpoint: String,
    },
}

/// File rotation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRotation {
    /// Maximum file size
    pub max_size: usize,
    /// Maximum file age
    pub max_age: Duration,
    /// Maximum file count
    pub max_files: usize,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Enable health checks
    pub enabled: bool,
    /// Check interval
    pub interval: Duration,
    /// Timeout per check
    pub timeout: Duration,
    /// Health check endpoints
    pub endpoints: Vec<HealthCheckEndpoint>,
}

/// Health check endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckEndpoint {
    /// Endpoint name
    pub name: String,
    /// Endpoint path
    pub path: String,
    /// Check type
    pub check_type: HealthCheckType,
}

/// Health check types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthCheckType {
    Liveness,
    Readiness,
    Startup,
}

// ─────────────────────────────────────────────────────────────
// SecurityConfig
// ─────────────────────────────────────────────────────────────

/// Security configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Authentication configuration
    pub authentication: AuthenticationConfig,
    /// Authorization configuration
    pub authorization: AuthorizationConfig,
    /// Encryption configuration
    pub encryption: EncryptionConfig,
    /// Audit configuration
    pub audit: AuditConfig,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
    /// Authentication providers
    pub providers: Vec<AuthProvider>,
    /// Session configuration
    pub session: SessionConfig,
    /// Token configuration
    pub token: TokenConfig,
}

/// Authentication providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthProvider {
    None,
    Basic {
        realm: String,
    },
    Bearer {
        issuer: String,
        audience: String,
    },
    OAuth2 {
        client_id: String,
        client_secret: String,
    },
    LDAP {
        server: String,
        base_dn: String,
    },
    Custom {
        provider_type: String,
        config: HashMap<String, String>,
    },
}

/// Session configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Session timeout
    pub timeout: Duration,
    /// Session storage
    pub storage: SessionStorage,
    /// Enable session encryption
    pub encryption: bool,
}

/// Session storage types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionStorage {
    Memory,
    File,
    Database,
    Redis,
}

/// Token configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenConfig {
    /// Token type
    pub token_type: TokenType,
    /// Token expiry
    pub expiry: Duration,
    /// Refresh token enabled
    pub refresh_enabled: bool,
    /// Token signing key
    pub signing_key: String,
}

/// Token types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenType {
    JWT,
    Opaque,
    PASETO,
}

/// Authorization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationConfig {
    /// Authorization model
    pub model: AuthorizationModel,
    /// Permissions
    pub permissions: Vec<Permission>,
    /// Roles
    pub roles: Vec<Role>,
    /// Policies
    pub policies: Vec<Policy>,
}

/// Authorization models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthorizationModel {
    None,
    RBAC,
    ABAC,
    Custom,
}

/// Permission definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    /// Permission name
    pub name: String,
    /// Description
    pub description: String,
    /// Resource type
    pub resource: String,
    /// Action
    pub action: String,
}

/// Role definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    /// Role name
    pub name: String,
    /// Description
    pub description: String,
    /// Permissions
    pub permissions: Vec<String>,
}

/// Policy definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// Policy name
    pub name: String,
    /// Policy expression
    pub expression: String,
    /// Effect (allow/deny)
    pub effect: PolicyEffect,
}

/// Policy effects
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyEffect {
    Allow,
    Deny,
}

/// Encryption configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Encryption at rest
    pub at_rest: EncryptionAtRest,
    /// Encryption in transit
    pub in_transit: EncryptionInTransit,
    /// Key management
    pub key_management: KeyManagementConfig,
}

/// Encryption at rest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionAtRest {
    /// Enable encryption
    pub enabled: bool,
    /// Encryption algorithm
    pub algorithm: EncryptionAlgorithm,
    /// Key size
    pub key_size: usize,
}

/// Encryption in transit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionInTransit {
    /// Enable TLS
    pub enabled: bool,
    /// TLS version
    pub tls_version: TlsVersion,
    /// Certificate path
    pub cert_path: PathBuf,
    /// Private key path
    pub key_path: PathBuf,
}

/// Encryption algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    AES128,
    AES256,
    ChaCha20,
    XChaCha20,
}

/// TLS versions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TlsVersion {
    TLSv1_2,
    TLSv1_3,
}

/// Key management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyManagementConfig {
    /// Key provider
    pub provider: KeyProvider,
    /// Key rotation interval
    pub rotation_interval: Duration,
    /// Key derivation function
    pub kdf: KeyDerivationFunction,
}

/// Key providers
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyProvider {
    File,
    Environment,
    HSM,
    Vault,
    AWS_KMS,
    Azure_KeyVault,
    GCP_KMS,
}

/// Key derivation functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyDerivationFunction {
    PBKDF2,
    Scrypt,
    Argon2,
}

/// Audit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Enable auditing
    pub enabled: bool,
    /// Audit log path
    pub log_path: PathBuf,
    /// Events to audit
    pub events: Vec<AuditEvent>,
    /// Audit log retention
    pub retention: Duration,
}

/// Audit events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditEvent {
    Authentication,
    Authorization,
    DataAccess,
    DataModification,
    AdminActions,
    SecurityEvents,
}

// ─────────────────────────────────────────────────────────────
// OptimizationConfig
// ─────────────────────────────────────────────────────────────

/// Advanced optimization configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// SIMD configuration
    pub simd: SimdConfig,
    /// Zero-copy configuration
    pub zero_copy: ZeroCopyConfig,
    /// Prefetching configuration
    pub prefetching: PrefetchingConfig,
    /// Caching configuration
    pub caching: crate::config_types_storage::CachingConfig,
}

/// SIMD configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdConfig {
    /// Enable SIMD operations
    pub enabled: bool,
    /// SIMD instruction set
    pub instruction_set: SimdInstructionSet,
    /// Fallback to scalar operations
    pub fallback_to_scalar: bool,
}

/// SIMD instruction sets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimdInstructionSet {
    None,
    SSE2,
    AVX,
    AVX2,
    AVX512,
    NEON,
    Auto,
}

/// Zero-copy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroCopyConfig {
    /// Enable zero-copy operations
    pub enabled: bool,
    /// Arena size for zero-copy allocations
    pub arena_size: usize,
    /// Enable reference counting
    pub reference_counting: bool,
}

/// Prefetching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchingConfig {
    /// Enable data prefetching
    pub enabled: bool,
    /// Prefetch distance
    pub distance: usize,
    /// Prefetch strategy
    pub strategy: PrefetchStrategy,
}

/// Prefetch strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrefetchStrategy {
    Sequential,
    Random,
    Adaptive,
}

// ─────────────────────────────────────────────────────────────
// Default implementations
// ─────────────────────────────────────────────────────────────

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            worker_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            stack_size: 2 * 1024 * 1024, // 2MB
            priority: ThreadPriority::Normal,
            work_stealing: true,
        }
    }
}

impl Default for LockConfig {
    fn default() -> Self {
        Self {
            default_type: LockType::RwLock,
            timeout: Duration::from_secs(30),
            enable_debugging: false,
            deadlock_detection: true,
        }
    }
}

impl Default for AsyncRuntimeConfig {
    fn default() -> Self {
        Self {
            runtime_type: AsyncRuntimeType::MultiThread,
            enable_io: true,
            enable_time: true,
            worker_config: AsyncWorkerConfig::default(),
        }
    }
}

impl Default for AsyncWorkerConfig {
    fn default() -> Self {
        Self {
            core_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            max_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                * 4,
            keep_alive: Duration::from_secs(60),
            thread_name_prefix: "oxirs-async".to_string(),
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            metrics: MetricsConfig::default(),
            tracing: TracingConfig::default(),
            logging: LoggingConfig::default(),
            health_checks: HealthCheckConfig::default(),
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            exporter: MetricsExporter::Prometheus,
            collection_interval: Duration::from_secs(15),
            retention_period: Duration::from_secs(3600),
            custom_metrics: Vec::new(),
        }
    }
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            exporter: TraceExporter::Jaeger,
            sampling_rate: 0.1,
            max_spans: 1000,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            format: LogFormat::Json,
            targets: vec![LogTarget::Console],
            structured: true,
        }
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            endpoints: Vec::new(),
        }
    }
}

impl Default for AuthenticationConfig {
    fn default() -> Self {
        Self {
            providers: vec![AuthProvider::None],
            session: SessionConfig::default(),
            token: TokenConfig::default(),
        }
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(3600),
            storage: SessionStorage::Memory,
            encryption: true,
        }
    }
}

impl Default for TokenConfig {
    fn default() -> Self {
        Self {
            token_type: TokenType::JWT,
            expiry: Duration::from_secs(3600),
            refresh_enabled: true,
            signing_key: "default_key".to_string(),
        }
    }
}

impl Default for AuthorizationConfig {
    fn default() -> Self {
        Self {
            model: AuthorizationModel::RBAC,
            permissions: Vec::new(),
            roles: Vec::new(),
            policies: Vec::new(),
        }
    }
}

impl Default for EncryptionAtRest {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: EncryptionAlgorithm::AES256,
            key_size: 256,
        }
    }
}

impl Default for EncryptionInTransit {
    fn default() -> Self {
        Self {
            enabled: true,
            tls_version: TlsVersion::TLSv1_3,
            cert_path: PathBuf::from("cert.pem"),
            key_path: PathBuf::from("key.pem"),
        }
    }
}

impl Default for KeyManagementConfig {
    fn default() -> Self {
        Self {
            provider: KeyProvider::File,
            rotation_interval: Duration::from_secs(86400 * 30), // 30 days
            kdf: KeyDerivationFunction::Argon2,
        }
    }
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            log_path: PathBuf::from("./audit.log"),
            events: vec![AuditEvent::Authentication, AuditEvent::Authorization],
            retention: Duration::from_secs(86400 * 365), // 1 year
        }
    }
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            instruction_set: SimdInstructionSet::Auto,
            fallback_to_scalar: true,
        }
    }
}

impl Default for ZeroCopyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            arena_size: 1024 * 1024, // 1MB
            reference_counting: true,
        }
    }
}

impl Default for PrefetchingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            distance: 64,
            strategy: PrefetchStrategy::Adaptive,
        }
    }
}
