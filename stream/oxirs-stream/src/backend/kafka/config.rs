//! Kafka Configuration Types
//!
//! This module contains all configuration-related types for the Kafka backend.

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

/// Partition strategies for message distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitionStrategy {
    RoundRobin,
    Hash,
    Manual,
    Sticky,
}

/// Workload profiles for performance optimization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkloadProfile {
    HighThroughput,
    LowLatency,
    Balanced,
    LargeMessages,
}

/// Performance configuration summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaPerformanceConfig {
    pub batch_size: u32,
    pub linger_ms: u32,
    pub buffer_memory: u64,
    pub compression_type: KafkaCompressionType,
    pub max_in_flight_requests: u32,
    pub request_timeout_ms: u32,
    pub delivery_timeout_ms: u32,
    pub retry_backoff_ms: u32,
    pub enable_idempotence: bool,
    pub acks: KafkaAcks,
}

/// Performance recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecommendation {
    pub category: String,
    pub description: String,
    pub suggested_action: String,
    pub impact: String,
}

/// Schema registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaRegistryConfig {
    pub url: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub api_key: Option<String>,
    pub timeout_ms: u32,
    pub cache_size: usize,
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
    pub ca_cert_file: Option<String>,
    pub cert_file: Option<String>,
    pub key_file: Option<String>,
    pub key_password: Option<String>,
    pub verify_hostname: bool,
}

/// Kafka consumer configuration with advanced features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaConsumerConfig {
    pub brokers: Vec<String>,
    pub group_id: String,
    pub client_id: String,
    pub auto_offset_reset: AutoOffsetReset,
    pub enable_auto_commit: bool,
    pub auto_commit_interval_ms: u32,
    pub session_timeout_ms: u32,
    pub heartbeat_interval_ms: u32,
    pub max_poll_interval_ms: u32,
    pub max_poll_records: u32,
    pub fetch_min_bytes: u32,
    pub fetch_max_wait_ms: u32,
    pub max_partition_fetch_bytes: u32,
    pub isolation_level: IsolationLevel,
    pub security_config: Option<KafkaSecurityConfig>,
    pub enable_metrics: bool,
    pub retry_config: Option<ConsumerRetryConfig>,
}

impl Default for KafkaConsumerConfig {
    fn default() -> Self {
        Self {
            brokers: vec!["localhost:9092".to_string()],
            group_id: format!("oxirs-group-{}", Uuid::new_v4()),
            client_id: format!("oxirs-consumer-{}", Uuid::new_v4()),
            auto_offset_reset: AutoOffsetReset::Latest,
            enable_auto_commit: false, // Manual commit for exactly-once
            auto_commit_interval_ms: 5000,
            session_timeout_ms: 30000,
            heartbeat_interval_ms: 3000,
            max_poll_interval_ms: 300000,
            max_poll_records: 500,
            fetch_min_bytes: 1,
            fetch_max_wait_ms: 500,
            max_partition_fetch_bytes: 1048576, // 1MB
            isolation_level: IsolationLevel::ReadCommitted,
            security_config: None,
            enable_metrics: true,
            retry_config: None,
        }
    }
}

/// Auto offset reset strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutoOffsetReset {
    Earliest,
    Latest,
    None,
}

impl ToString for AutoOffsetReset {
    fn to_string(&self) -> String {
        match self {
            AutoOffsetReset::Earliest => "earliest".to_string(),
            AutoOffsetReset::Latest => "latest".to_string(),
            AutoOffsetReset::None => "none".to_string(),
        }
    }
}

/// Isolation levels for transactional reads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
}

impl ToString for IsolationLevel {
    fn to_string(&self) -> String {
        match self {
            IsolationLevel::ReadUncommitted => "read_uncommitted".to_string(),
            IsolationLevel::ReadCommitted => "read_committed".to_string(),
        }
    }
}

/// Consumer retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumerRetryConfig {
    pub max_retries: u32,
    pub retry_delay_ms: u32,
    pub backoff_multiplier: f64,
    pub max_retry_delay_ms: u32,
}

impl Default for ConsumerRetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_delay_ms: 1000,
            backoff_multiplier: 2.0,
            max_retry_delay_ms: 30000,
        }
    }
}

impl WorkloadProfile {
    /// Get optimized configuration for a specific workload profile
    pub fn get_performance_config(&self) -> KafkaPerformanceConfig {
        match self {
            WorkloadProfile::HighThroughput => KafkaPerformanceConfig {
                batch_size: 131072,  // 128KB
                linger_ms: 50,
                buffer_memory: 67108864, // 64MB
                compression_type: KafkaCompressionType::Lz4,
                max_in_flight_requests: 10,
                request_timeout_ms: 60000,
                delivery_timeout_ms: 600000,
                retry_backoff_ms: 500,
                enable_idempotence: true,
                acks: KafkaAcks::All,
            },
            WorkloadProfile::LowLatency => KafkaPerformanceConfig {
                batch_size: 16384,   // 16KB
                linger_ms: 0,
                buffer_memory: 16777216, // 16MB
                compression_type: KafkaCompressionType::None,
                max_in_flight_requests: 1,
                request_timeout_ms: 5000,
                delivery_timeout_ms: 30000,
                retry_backoff_ms: 50,
                enable_idempotence: false,
                acks: KafkaAcks::Leader,
            },
            WorkloadProfile::Balanced => KafkaPerformanceConfig {
                batch_size: 65536,   // 64KB
                linger_ms: 10,
                buffer_memory: 33554432, // 32MB
                compression_type: KafkaCompressionType::Snappy,
                max_in_flight_requests: 5,
                request_timeout_ms: 30000,
                delivery_timeout_ms: 300000,
                retry_backoff_ms: 100,
                enable_idempotence: true,
                acks: KafkaAcks::All,
            },
            WorkloadProfile::LargeMessages => KafkaPerformanceConfig {
                batch_size: 262144,  // 256KB
                linger_ms: 100,
                buffer_memory: 134217728, // 128MB
                compression_type: KafkaCompressionType::Zstd,
                max_in_flight_requests: 3,
                request_timeout_ms: 120000,
                delivery_timeout_ms: 1200000,
                retry_backoff_ms: 1000,
                enable_idempotence: true,
                acks: KafkaAcks::All,
            },
        }
    }

    /// Get performance recommendations for the current configuration
    pub fn analyze_configuration(&self, current: &KafkaPerformanceConfig) -> Vec<PerformanceRecommendation> {
        let optimal = self.get_performance_config();
        let mut recommendations = Vec::new();

        if current.batch_size < optimal.batch_size / 2 {
            recommendations.push(PerformanceRecommendation {
                category: "Throughput".to_string(),
                description: "Batch size is significantly smaller than recommended".to_string(),
                suggested_action: format!("Increase batch_size to {}", optimal.batch_size),
                impact: "High throughput improvement".to_string(),
            });
        }

        if current.linger_ms > optimal.linger_ms * 2 {
            recommendations.push(PerformanceRecommendation {
                category: "Latency".to_string(),
                description: "Linger time is higher than recommended for this workload".to_string(),
                suggested_action: format!("Reduce linger_ms to {}", optimal.linger_ms),
                impact: "Lower latency".to_string(),
            });
        }

        recommendations
    }
}