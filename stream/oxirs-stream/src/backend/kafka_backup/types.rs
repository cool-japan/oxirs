//! Kafka configuration types and enums

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

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
    pub security_protocol: SecurityProtocol,
    pub sasl_mechanism: Option<SaslMechanism>,
    pub sasl_username: Option<String>,
    pub sasl_password: Option<String>,
    pub ssl_ca_location: Option<String>,
    pub ssl_certificate_location: Option<String>,
    pub ssl_key_location: Option<String>,
    pub ssl_key_password: Option<String>,
    pub enable_ssl_certificate_verification: bool,
    pub enable_partitioner_consistency: bool,
    pub max_in_flight_requests_per_connection: u32,
    pub delivery_timeout_ms: u32,
    pub request_timeout_ms: u32,
    pub metadata_max_age_ms: u32,
    pub topic: String,
    pub partition_strategy: PartitionStrategy,
    pub custom_headers: HashMap<String, String>,
    pub schema_registry_url: Option<String>,
    pub schema_registry_auth: Option<SchemaRegistryAuth>,
}

/// Kafka acknowledgment levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KafkaAcks {
    None,      // 0: No acknowledgment
    Leader,    // 1: Leader acknowledgment only
    All,       // -1: All in-sync replicas
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

/// Security protocols for Kafka
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityProtocol {
    Plaintext,
    Ssl,
    SaslPlaintext,
    SaslSsl,
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

/// Partition strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitionStrategy {
    RoundRobin,
    Hash,
    Random,
    Sticky,
    Custom(String),
}

/// Schema registry authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaRegistryAuth {
    pub username: String,
    pub password: String,
}

/// Kafka consumer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaConsumerConfig {
    pub brokers: Vec<String>,
    pub group_id: String,
    pub client_id: String,
    pub enable_auto_commit: bool,
    pub auto_commit_interval_ms: u32,
    pub auto_offset_reset: AutoOffsetReset,
    pub isolation_level: IsolationLevel,
    pub max_poll_records: u32,
    pub fetch_min_bytes: u32,
    pub fetch_max_wait_ms: u32,
    pub max_partition_fetch_bytes: u32,
    pub session_timeout_ms: u32,
    pub heartbeat_interval_ms: u32,
    pub security_protocol: SecurityProtocol,
    pub sasl_mechanism: Option<SaslMechanism>,
    pub sasl_username: Option<String>,
    pub sasl_password: Option<String>,
    pub topics: Vec<String>,
    pub schema_registry_url: Option<String>,
    pub schema_registry_auth: Option<SchemaRegistryAuth>,
}

/// Auto offset reset strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutoOffsetReset {
    Earliest,
    Latest,
    None,
}

/// Isolation levels for transactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
}

impl Default for KafkaProducerConfig {
    fn default() -> Self {
        Self {
            brokers: vec!["localhost:9092".to_string()],
            client_id: "oxirs-producer".to_string(),
            transaction_id: None,
            enable_idempotence: true,
            acks: KafkaAcks::All,
            retries: 2147483647,
            retry_backoff_ms: 100,
            batch_size: 16384,
            linger_ms: 5,
            buffer_memory: 33554432,
            compression_type: KafkaCompressionType::Lz4,
            security_protocol: SecurityProtocol::Plaintext,
            sasl_mechanism: None,
            sasl_username: None,
            sasl_password: None,
            ssl_ca_location: None,
            ssl_certificate_location: None,
            ssl_key_location: None,
            ssl_key_password: None,
            enable_ssl_certificate_verification: true,
            enable_partitioner_consistency: true,
            max_in_flight_requests_per_connection: 5,
            delivery_timeout_ms: 120000,
            request_timeout_ms: 30000,
            metadata_max_age_ms: 300000,
            topic: "rdf-events".to_string(),
            partition_strategy: PartitionStrategy::Hash,
            custom_headers: HashMap::new(),
            schema_registry_url: None,
            schema_registry_auth: None,
        }
    }
}

impl Default for KafkaConsumerConfig {
    fn default() -> Self {
        Self {
            brokers: vec!["localhost:9092".to_string()],
            group_id: "oxirs-consumer-group".to_string(),
            client_id: "oxirs-consumer".to_string(),
            enable_auto_commit: false,
            auto_commit_interval_ms: 5000,
            auto_offset_reset: AutoOffsetReset::Latest,
            isolation_level: IsolationLevel::ReadCommitted,
            max_poll_records: 500,
            fetch_min_bytes: 1,
            fetch_max_wait_ms: 500,
            max_partition_fetch_bytes: 1048576,
            session_timeout_ms: 30000,
            heartbeat_interval_ms: 3000,
            security_protocol: SecurityProtocol::Plaintext,
            sasl_mechanism: None,
            sasl_username: None,
            sasl_password: None,
            topics: vec!["rdf-events".to_string()],
            schema_registry_url: None,
            schema_registry_auth: None,
        }
    }
}