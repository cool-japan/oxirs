//! Configuration types for stream producer/consumer

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Compression types supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionType {
    None,
    Gzip,
    Snappy,
    Lz4,
    Zstd,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    pub backoff_multiplier: f64,
    pub jitter: bool,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub enabled: bool,
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout: Duration,
    pub half_open_max_calls: u32,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub enable_tls: bool,
    pub verify_certificates: bool,
    pub client_cert_path: Option<String>,
    pub client_key_path: Option<String>,
    pub ca_cert_path: Option<String>,
    pub sasl_config: Option<SaslConfig>,
}

/// SASL authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaslConfig {
    pub mechanism: SaslMechanism,
    pub username: String,
    pub password: String,
}

/// SASL authentication mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SaslMechanism {
    Plain,
    ScramSha256,
    ScramSha512,
    OAuthBearer,
}

/// Performance tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamPerformanceConfig {
    pub enable_batching: bool,
    pub enable_pipelining: bool,
    pub buffer_size: usize,
    pub prefetch_count: u32,
    pub enable_zero_copy: bool,
    pub enable_simd: bool,
    pub parallel_processing: bool,
    pub worker_threads: Option<usize>,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_metrics: bool,
    pub enable_tracing: bool,
    pub metrics_interval: Duration,
    pub health_check_interval: Duration,
    pub enable_profiling: bool,
    pub prometheus_endpoint: Option<String>,
    pub otlp_endpoint: Option<String>,
    pub log_level: String,
}

/// NATS JetStream configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NatsJetStreamConfig {
    pub domain: Option<String>,
    pub api_prefix: Option<String>,
    pub timeout: Duration,
}

/// AWS credentials configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwsCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
    pub role_arn: Option<String>,
}

/// Pulsar authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PulsarAuthConfig {
    pub auth_method: PulsarAuthMethod,
    pub auth_params: HashMap<String, String>,
}

/// Pulsar authentication methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PulsarAuthMethod {
    Token,
    Jwt,
    Oauth2,
    Tls,
}

/// Enhanced streaming backend options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamBackendType {
    /// Apache Kafka backend selector. The variant (pure config data) stays here so
    /// configs remain API-stable, but the `rdkafka`-backed implementation was
    /// quarantined into the publish=false `oxirs-stream-adapter-rdkafka` crate
    /// (Pure Rust Policy v2). Build a `KafkaBackend` from that crate via the
    /// `StreamBackend` trait; `StreamProducer::new`/`StreamConsumer::new` return an
    /// error explaining this for a `Kafka` config.
    Kafka {
        brokers: Vec<String>,
        security_protocol: Option<String>,
        sasl_config: Option<SaslConfig>,
    },
    #[cfg(feature = "nats")]
    Nats {
        url: String,
        cluster_urls: Option<Vec<String>>,
        jetstream_config: Option<NatsJetStreamConfig>,
    },
    #[cfg(feature = "redis")]
    Redis {
        url: String,
        cluster_urls: Option<Vec<String>>,
        pool_size: Option<usize>,
    },
    #[cfg(feature = "kinesis")]
    Kinesis {
        region: String,
        stream_name: String,
        credentials: Option<AwsCredentials>,
    },
    /// Apache Pulsar backend selector. The variant (pure config data) stays here so
    /// configs remain API-stable, but the `pulsar`-backed implementation was
    /// quarantined into the publish=false `oxirs-stream-adapter-pulsar` crate
    /// (Pure Rust Policy v2). Build a `PulsarProducer`/`PulsarConsumer` from that
    /// crate directly; `StreamProducer::new`/`StreamConsumer::new` return an error
    /// explaining this for a `Pulsar` config.
    Pulsar {
        service_url: String,
        auth_config: Option<PulsarAuthConfig>,
    },
    #[cfg(feature = "rabbitmq")]
    RabbitMQ {
        url: String,
        exchange: Option<String>,
        queue: Option<String>,
    },
    Memory {
        max_size: Option<usize>,
        persistence: bool,
    },
}

/// Enhanced stream configuration with advanced features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    pub backend: StreamBackendType,
    pub topic: String,
    pub batch_size: usize,
    pub flush_interval_ms: u64,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Enable compression
    pub enable_compression: bool,
    /// Compression type
    pub compression_type: CompressionType,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
    /// Security configuration
    pub security: SecurityConfig,
    /// Performance tuning
    pub performance: StreamPerformanceConfig,
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
}

// ============================================================
// Default implementations
// ============================================================

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(30),
            half_open_max_calls: 3,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_tls: false,
            verify_certificates: true,
            client_cert_path: None,
            client_key_path: None,
            ca_cert_path: None,
            sasl_config: None,
        }
    }
}

impl Default for StreamPerformanceConfig {
    fn default() -> Self {
        Self {
            enable_batching: true,
            enable_pipelining: false,
            buffer_size: 8192,
            prefetch_count: 100,
            enable_zero_copy: false,
            enable_simd: false,
            parallel_processing: true,
            worker_threads: None,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            enable_tracing: true,
            metrics_interval: Duration::from_secs(60),
            health_check_interval: Duration::from_secs(30),
            enable_profiling: false,
            prometheus_endpoint: None,
            otlp_endpoint: None,
            log_level: "info".to_string(),
        }
    }
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            backend: StreamBackendType::Memory {
                max_size: Some(10000),
                persistence: false,
            },
            topic: "oxirs-stream".to_string(),
            batch_size: 100,
            flush_interval_ms: 100,
            max_connections: 10,
            connection_timeout: Duration::from_secs(30),
            enable_compression: false,
            compression_type: CompressionType::None,
            retry_config: RetryConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            security: SecurityConfig::default(),
            performance: StreamPerformanceConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

/// Helper functions for creating common configurations
impl StreamConfig {
    /// Create a Redis configuration
    #[cfg(feature = "redis")]
    pub fn redis(url: String) -> Self {
        Self {
            backend: StreamBackendType::Redis {
                url,
                cluster_urls: None,
                pool_size: Some(10),
            },
            ..Default::default()
        }
    }

    /// Create a Kinesis configuration
    #[cfg(feature = "kinesis")]
    pub fn kinesis(region: String, stream_name: String) -> Self {
        Self {
            backend: StreamBackendType::Kinesis {
                region,
                stream_name,
                credentials: None,
            },
            ..Default::default()
        }
    }

    /// Create a memory configuration for testing
    pub fn memory() -> Self {
        Self {
            backend: StreamBackendType::Memory {
                max_size: Some(1000),
                persistence: false,
            },
            ..Default::default()
        }
    }

    /// Enable high-performance configuration
    pub fn high_performance(mut self) -> Self {
        self.performance.enable_batching = true;
        self.performance.enable_pipelining = true;
        self.performance.parallel_processing = true;
        self.performance.buffer_size = 65536;
        self.performance.prefetch_count = 1000;
        self.batch_size = 1000;
        self.flush_interval_ms = 10;
        self
    }

    /// Enable compression
    pub fn with_compression(mut self, compression_type: CompressionType) -> Self {
        self.enable_compression = true;
        self.compression_type = compression_type;
        self
    }

    /// Configure circuit breaker
    pub fn with_circuit_breaker(mut self, enabled: bool, failure_threshold: u32) -> Self {
        self.circuit_breaker.enabled = enabled;
        self.circuit_breaker.failure_threshold = failure_threshold;
        self
    }

    /// Create a development configuration with memory backend and debug settings
    pub fn development(topic: &str) -> Self {
        Self {
            backend: StreamBackendType::Memory {
                max_size: Some(10000),
                persistence: false,
            },
            topic: topic.to_string(),
            batch_size: 10,
            flush_interval_ms: 100,
            max_connections: 5,
            connection_timeout: Duration::from_secs(10),
            enable_compression: false,
            compression_type: CompressionType::None,
            retry_config: RetryConfig {
                max_retries: 3,
                initial_backoff: Duration::from_millis(100),
                max_backoff: Duration::from_secs(5),
                backoff_multiplier: 2.0,
                jitter: true,
            },
            circuit_breaker: CircuitBreakerConfig {
                enabled: false,
                failure_threshold: 5,
                success_threshold: 2,
                timeout: Duration::from_secs(60),
                half_open_max_calls: 10,
            },
            security: SecurityConfig::default(),
            performance: StreamPerformanceConfig::default(),
            monitoring: MonitoringConfig {
                enable_metrics: true,
                enable_tracing: false,
                metrics_interval: Duration::from_secs(5),
                health_check_interval: Duration::from_secs(30),
                enable_profiling: false,
                prometheus_endpoint: None,
                otlp_endpoint: None,
                log_level: "debug".to_string(),
            },
        }
    }

    /// Create a production configuration with optimal performance settings
    pub fn production(topic: &str) -> Self {
        Self {
            backend: StreamBackendType::Memory {
                max_size: Some(100000),
                persistence: true,
            },
            topic: topic.to_string(),
            batch_size: 1000,
            flush_interval_ms: 10,
            max_connections: 50,
            connection_timeout: Duration::from_secs(30),
            enable_compression: true,
            compression_type: CompressionType::Zstd,
            retry_config: RetryConfig {
                max_retries: 5,
                initial_backoff: Duration::from_millis(200),
                max_backoff: Duration::from_secs(30),
                backoff_multiplier: 2.0,
                jitter: true,
            },
            circuit_breaker: CircuitBreakerConfig {
                enabled: true,
                failure_threshold: 10,
                success_threshold: 3,
                timeout: Duration::from_secs(30),
                half_open_max_calls: 5,
            },
            security: SecurityConfig::default(),
            performance: StreamPerformanceConfig {
                enable_batching: true,
                enable_pipelining: true,
                parallel_processing: true,
                buffer_size: 65536,
                prefetch_count: 1000,
                enable_zero_copy: true,
                enable_simd: true,
                worker_threads: None,
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                enable_tracing: true,
                metrics_interval: Duration::from_secs(1),
                health_check_interval: Duration::from_secs(10),
                enable_profiling: true,
                prometheus_endpoint: None,
                otlp_endpoint: None,
                log_level: "info".to_string(),
            },
        }
    }
}
