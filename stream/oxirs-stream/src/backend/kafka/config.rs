//! Kafka configuration types and settings

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Enhanced Kafka producer configuration with enterprise features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaProducerConfig {
    pub brokers: Vec<String>,
    pub client_id: String,
    pub transaction_id: Option<String>,
    pub enable_idempotence: bool,
    pub acks: KafkaAcks,
    pub retries: u32,
    pub retry_backoff_ms: u32,
    pub batch_size: u32,
    pub linger_ms: u32,
    pub buffer_memory: u64,
    pub compression_type: KafkaCompressionType,
    pub max_in_flight_requests: u32,
    pub request_timeout_ms: u32,
    pub delivery_timeout_ms: u32,
    pub enable_metrics: bool,
    pub schema_registry_config: Option<SchemaRegistryConfig>,
    pub security_config: Option<KafkaSecurityConfig>,
    pub partition_strategy: PartitionStrategy,
    pub headers: HashMap<String, String>,
}

impl Default for KafkaProducerConfig {
    fn default() -> Self {
        Self {
            brokers: vec!["localhost:9092".to_string()],
            client_id: format!("oxirs-producer-{}", Uuid::new_v4()),
            transaction_id: None,
            enable_idempotence: true,
            acks: KafkaAcks::All,
            retries: 2147483647, // Max retries for exactly-once
            retry_backoff_ms: 100,
            batch_size: 65536,
            linger_ms: 10,
            buffer_memory: 33554432, // 32MB
            compression_type: KafkaCompressionType::Snappy,
            max_in_flight_requests: 5,
            request_timeout_ms: 30000,
            delivery_timeout_ms: 300000,
            enable_metrics: true,
            schema_registry_config: None,
            security_config: None,
            partition_strategy: PartitionStrategy::RoundRobin,
            headers: HashMap::new(),
        }
    }
}

/// Kafka acknowledgment levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KafkaAcks {
    None,
    Leader,
    All,
}

impl std::fmt::Display for KafkaAcks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KafkaAcks::None => write!(f, "0"),
            KafkaAcks::Leader => write!(f, "1"),
            KafkaAcks::All => write!(f, "all"),
        }
    }
}

/// Kafka compression types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KafkaCompressionType {
    None,
    Gzip,
    Snappy,
    Lz4,
    Zstd,
}

impl std::fmt::Display for KafkaCompressionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KafkaCompressionType::None => write!(f, "none"),
            KafkaCompressionType::Gzip => write!(f, "gzip"),
            KafkaCompressionType::Snappy => write!(f, "snappy"),
            KafkaCompressionType::Lz4 => write!(f, "lz4"),
            KafkaCompressionType::Zstd => write!(f, "zstd"),
        }
    }
}

/// Partition strategies for message distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitionStrategy {
    RoundRobin,
    Hash,
    Manual,
    Sticky,
}

/// Schema Registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaRegistryConfig {
    pub url: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub timeout_ms: u32,
    pub cache_size: usize,
}

impl Default for SchemaRegistryConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:8081".to_string(),
            username: None,
            password: None,
            timeout_ms: 30000,
            cache_size: 1000,
        }
    }
}

/// Kafka security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaSecurityConfig {
    pub security_protocol: SecurityProtocol,
    pub sasl_config: Option<SaslConfig>,
    pub ssl_config: Option<SslConfig>,
}

/// Security protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityProtocol {
    Plaintext,
    Ssl,
    SaslPlaintext,
    SaslSsl,
}

impl std::fmt::Display for SecurityProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SecurityProtocol::Plaintext => write!(f, "PLAINTEXT"),
            SecurityProtocol::Ssl => write!(f, "SSL"),
            SecurityProtocol::SaslPlaintext => write!(f, "SASL_PLAINTEXT"),
            SecurityProtocol::SaslSsl => write!(f, "SASL_SSL"),
        }
    }
}

/// SASL authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaslConfig {
    pub mechanism: SaslMechanism,
    pub username: String,
    pub password: String,
}

/// SASL mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SaslMechanism {
    Plain,
    ScramSha256,
    ScramSha512,
    Gssapi,
    OAuthBearer,
}

impl std::fmt::Display for SaslMechanism {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SaslMechanism::Plain => write!(f, "PLAIN"),
            SaslMechanism::ScramSha256 => write!(f, "SCRAM-SHA-256"),
            SaslMechanism::ScramSha512 => write!(f, "SCRAM-SHA-512"),
            SaslMechanism::Gssapi => write!(f, "GSSAPI"),
            SaslMechanism::OAuthBearer => write!(f, "OAUTHBEARER"),
        }
    }
}

/// SSL configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SslConfig {
    pub ca_location: Option<String>,
    pub certificate_location: Option<String>,
    pub key_location: Option<String>,
    pub key_password: Option<String>,
    pub keystore_location: Option<String>,
    pub keystore_password: Option<String>,
}

/// Producer statistics for monitoring
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KafkaProducerStats {
    pub events_published: u64,
    pub events_failed: u64,
    pub bytes_sent: u64,
    pub delivery_errors: u64,
    pub last_publish: Option<chrono::DateTime<chrono::Utc>>,
    pub max_latency_ms: u64,
    pub avg_latency_ms: f64,
}
