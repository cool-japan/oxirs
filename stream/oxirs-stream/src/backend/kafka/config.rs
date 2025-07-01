//! Kafka configuration types and utilities

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
            retries: 2147483647,
            retry_backoff_ms: 100,  
            batch_size: 65536,
            linger_ms: 10,
            buffer_memory: 33554432,
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

impl ToString for KafkaAcks {
    fn to_string(&self) -> String {
        match self {
            KafkaAcks::None => "0".to_string(),
            KafkaAcks::Leader => "1".to_string(), 
            KafkaAcks::All => "all".to_string(),
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

impl ToString for KafkaCompressionType {
    fn to_string(&self) -> String {
        match self {
            KafkaCompressionType::None => "none".to_string(),
            KafkaCompressionType::Gzip => "gzip".to_string(),
            KafkaCompressionType::Snappy => "snappy".to_string(),
            KafkaCompressionType::Lz4 => "lz4".to_string(),
            KafkaCompressionType::Zstd => "zstd".to_string(),
        }
    }
}

/// Partition strategy for message routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitionStrategy {
    RoundRobin,
    Random,
    KeyBased,
    Manual,
}

/// Schema registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaRegistryConfig {
    pub url: String,
    pub auth: Option<SchemaRegistryAuth>,
    pub timeout_ms: u32,
    pub cache_size: usize,
}

/// Schema registry authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaRegistryAuth {
    pub username: String,
    pub password: String,
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

impl ToString for SecurityProtocol {
    fn to_string(&self) -> String {
        match self {
            SecurityProtocol::Plaintext => "PLAINTEXT".to_string(),
            SecurityProtocol::Ssl => "SSL".to_string(),
            SecurityProtocol::SaslPlaintext => "SASL_PLAINTEXT".to_string(),
            SecurityProtocol::SaslSsl => "SASL_SSL".to_string(),
        }
    }
}

/// SASL configuration
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

impl ToString for SaslMechanism {
    fn to_string(&self) -> String {
        match self {
            SaslMechanism::Plain => "PLAIN".to_string(),
            SaslMechanism::ScramSha256 => "SCRAM-SHA-256".to_string(),
            SaslMechanism::ScramSha512 => "SCRAM-SHA-512".to_string(),
            SaslMechanism::Gssapi => "GSSAPI".to_string(),
            SaslMechanism::OAuthBearer => "OAUTHBEARER".to_string(),
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