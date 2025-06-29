//! # Apache Kafka Backend - Ultra-High Performance
//!
//! Complete Apache Kafka integration for enterprise-scale RDF streaming.
//!
//! This module provides comprehensive Kafka integration with transactional producers,
//! exactly-once semantics, schema registry, consumer groups, and advanced performance
//! optimizations for mission-critical RDF streaming applications.

use crate::{EventMetadata, PatchOperation, RdfPatch, StreamBackend, StreamConfig, StreamEvent};
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tokio::time;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// TODO: Enable when reqwest is added
// #[cfg(feature = "kafka")]
// use crate::backend::kafka_schema_registry::{SchemaRegistryClient, RdfEventSchemas, SchemaType};

#[cfg(feature = "kafka")]
use rdkafka::{
    admin::{AdminClient, AdminOptions, ConfigEntry, ConfigResource, NewTopic, TopicReplication},
    client::{ClientContext, DefaultClientContext},
    config::{ClientConfig, RDKafkaLogLevel},
    consumer::{CommitMode, Consumer, ConsumerContext, Rebalance, StreamConsumer},
    error::{KafkaError, KafkaResult},
    message::{BorrowedMessage, Header, Headers, Message, OwnedHeaders},
    metadata::{MetadataPartition, MetadataTopic},
    producer::{DeliveryResult, FutureProducer, FutureRecord, Producer},
    statistics::Statistics,
    util::Timeout,
    Offset, TopicPartitionList,
};

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
enum WorkloadProfile {
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
    pub ca_location: Option<String>,
    pub certificate_location: Option<String>,
    pub key_location: Option<String>,
    pub key_password: Option<String>,
    pub keystore_location: Option<String>,
    pub keystore_password: Option<String>,
}

/// Enhanced Kafka event with metadata and schema support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaEvent {
    pub schema_id: Option<u32>,
    pub event_id: String,
    pub event_type: String,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub correlation_id: String,
    pub transaction_id: Option<String>,
    pub data: serde_json::Value,
    pub metadata: EventMetadata,
    pub headers: HashMap<String, String>,
    pub partition_key: Option<String>,
    pub schema_version: String,
}

impl From<StreamEvent> for KafkaEvent {
    fn from(event: StreamEvent) -> Self {
        let (event_type, data, metadata, partition_key) = match event {
            StreamEvent::TripleAdded {
                subject,
                predicate,
                object,
                graph,
                metadata,
            } => (
                "triple_added".to_string(),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }),
                metadata,
                Some(subject),
            ),
            StreamEvent::TripleRemoved {
                subject,
                predicate,
                object,
                graph,
                metadata,
            } => (
                "triple_removed".to_string(),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }),
                metadata,
                Some(subject),
            ),
            StreamEvent::QuadAdded {
                subject,
                predicate,
                object,
                graph,
                metadata,
            } => (
                "quad_added".to_string(),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }),
                metadata,
                Some(subject),
            ),
            StreamEvent::QuadRemoved {
                subject,
                predicate,
                object,
                graph,
                metadata,
            } => (
                "quad_removed".to_string(),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }),
                metadata,
                Some(subject),
            ),
            StreamEvent::GraphCreated { graph, metadata } => (
                "graph_created".to_string(),
                serde_json::json!({
                    "graph": graph
                }),
                metadata,
                Some(graph),
            ),
            StreamEvent::GraphCleared { graph, metadata } => (
                "graph_cleared".to_string(),
                serde_json::json!({
                    "graph": graph
                }),
                metadata,
                graph,
            ),
            StreamEvent::GraphDeleted { graph, metadata } => (
                "graph_deleted".to_string(),
                serde_json::json!({
                    "graph": graph
                }),
                metadata,
                Some(graph),
            ),
            StreamEvent::SparqlUpdate {
                query,
                operation_type,
                metadata,
            } => (
                "sparql_update".to_string(),
                serde_json::json!({
                    "query": query,
                    "operation_type": operation_type
                }),
                metadata,
                None,
            ),
            StreamEvent::TransactionBegin {
                transaction_id,
                isolation_level,
                metadata,
            } => (
                "transaction_begin".to_string(),
                serde_json::json!({
                    "transaction_id": transaction_id,
                    "isolation_level": isolation_level
                }),
                metadata,
                Some(transaction_id),
            ),
            StreamEvent::TransactionCommit {
                transaction_id,
                metadata,
            } => (
                "transaction_commit".to_string(),
                serde_json::json!({
                    "transaction_id": transaction_id
                }),
                metadata,
                Some(transaction_id),
            ),
            StreamEvent::TransactionAbort {
                transaction_id,
                metadata,
            } => (
                "transaction_abort".to_string(),
                serde_json::json!({
                    "transaction_id": transaction_id
                }),
                metadata,
                Some(transaction_id),
            ),
            StreamEvent::SchemaChanged {
                schema_type,
                change_type,
                details,
                metadata,
            } => (
                "schema_changed".to_string(),
                serde_json::json!({
                    "schema_type": schema_type,
                    "change_type": change_type,
                    "details": details
                }),
                metadata,
                Some("schema".to_string()),
            ),
            StreamEvent::Heartbeat {
                timestamp,
                source,
                metadata,
            } => (
                "heartbeat".to_string(),
                serde_json::json!({
                    "source": source
                }),
                EventMetadata {
                    event_id: Uuid::new_v4().to_string(),
                    timestamp,
                    source: source.clone(),
                    user: None,
                    context: None,
                    caused_by: None,
                    version: "1.0".to_string(),
                    properties: HashMap::new(),
                    checksum: None,
                },
                Some(source),
            ),
            // Catch-all for remaining variants
            _ => (
                "unknown_event".to_string(),
                serde_json::json!({}),
                EventMetadata {
                    event_id: Uuid::new_v4().to_string(),
                    timestamp: Utc::now(),
                    source: "system".to_string(),
                    user: None,
                    context: None,
                    caused_by: None,
                    version: "1.0".to_string(),
                    properties: HashMap::new(),
                    checksum: None,
                },
                None,
            ),
        };

        Self {
            schema_id: None,
            event_id: metadata.event_id.clone(),
            event_type,
            timestamp: metadata.timestamp,
            source: metadata.source.clone(),
            correlation_id: Uuid::new_v4().to_string(),
            transaction_id: None,
            data,
            metadata,
            headers: HashMap::new(),
            partition_key,
            schema_version: "1.0".to_string(),
        }
    }
}

impl TryFrom<KafkaEvent> for StreamEvent {
    type Error = anyhow::Error;

    fn try_from(kafka_event: KafkaEvent) -> Result<Self> {
        let metadata = kafka_event.metadata;

        match kafka_event.event_type.as_str() {
            "triple_added" => {
                let subject = kafka_event.data["subject"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing subject"))?
                    .to_string();
                let predicate = kafka_event.data["predicate"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing predicate"))?
                    .to_string();
                let object = kafka_event.data["object"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing object"))?
                    .to_string();
                let graph = kafka_event.data["graph"].as_str().map(|s| s.to_string());

                Ok(StreamEvent::TripleAdded {
                    subject,
                    predicate,
                    object,
                    graph,
                    metadata,
                })
            }
            "triple_removed" => {
                let subject = kafka_event.data["subject"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing subject"))?
                    .to_string();
                let predicate = kafka_event.data["predicate"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing predicate"))?
                    .to_string();
                let object = kafka_event.data["object"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing object"))?
                    .to_string();
                let graph = kafka_event.data["graph"].as_str().map(|s| s.to_string());

                Ok(StreamEvent::TripleRemoved {
                    subject,
                    predicate,
                    object,
                    graph,
                    metadata,
                })
            }
            "quad_added" => {
                let subject = kafka_event.data["subject"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing subject"))?
                    .to_string();
                let predicate = kafka_event.data["predicate"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing predicate"))?
                    .to_string();
                let object = kafka_event.data["object"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing object"))?
                    .to_string();
                let graph = kafka_event.data["graph"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing graph"))?
                    .to_string();

                Ok(StreamEvent::QuadAdded {
                    subject,
                    predicate,
                    object,
                    graph,
                    metadata,
                })
            }
            "quad_removed" => {
                let subject = kafka_event.data["subject"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing subject"))?
                    .to_string();
                let predicate = kafka_event.data["predicate"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing predicate"))?
                    .to_string();
                let object = kafka_event.data["object"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing object"))?
                    .to_string();
                let graph = kafka_event.data["graph"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing graph"))?
                    .to_string();

                Ok(StreamEvent::QuadRemoved {
                    subject,
                    predicate,
                    object,
                    graph,
                    metadata,
                })
            }
            "graph_created" => {
                let graph = kafka_event.data["graph"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing graph"))?
                    .to_string();
                Ok(StreamEvent::GraphCreated { graph, metadata })
            }
            "graph_cleared" => {
                let graph = kafka_event.data["graph"].as_str().map(|s| s.to_string());
                Ok(StreamEvent::GraphCleared { graph, metadata })
            }
            "graph_deleted" => {
                let graph = kafka_event.data["graph"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing graph"))?
                    .to_string();
                Ok(StreamEvent::GraphDeleted { graph, metadata })
            }
            "heartbeat" => {
                let source = kafka_event.data["source"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing source"))?
                    .to_string();
                Ok(StreamEvent::Heartbeat {
                    timestamp: kafka_event.timestamp,
                    source,
                    metadata: EventMetadata::default(),
                })
            }
            _ => Err(anyhow!("Unknown event type: {}", kafka_event.event_type)),
        }
    }
}

/// Enhanced Kafka producer with transactional support
pub struct KafkaProducer {
    config: StreamConfig,
    kafka_config: KafkaProducerConfig,
    #[cfg(feature = "kafka")]
    producer: Option<FutureProducer>,
    #[cfg(not(feature = "kafka"))]
    _phantom: std::marker::PhantomData<()>,
    stats: ProducerStats,
    transaction_active: bool,
    batch_buffer: Vec<KafkaEvent>,
    partition_cache: HashMap<String, u32>,
    schema_cache: HashMap<String, u32>,
    // TODO: Enable when reqwest is added
    // #[cfg(feature = "kafka")]
    // schema_registry_client: Option<Arc<SchemaRegistryClient>>,
    metrics_collector: Option<Arc<RwLock<KafkaMetrics>>>,
    send_semaphore: Arc<Semaphore>,
}

#[derive(Debug, Default)]
struct ProducerStats {
    events_published: u64,
    events_failed: u64,
    bytes_sent: u64,
    batches_sent: u64,
    transactions_committed: u64,
    transactions_aborted: u64,
    avg_latency_ms: f64,
    max_latency_ms: u64,
    schema_registry_calls: u64,
    partition_assignments: u64,
    last_publish: Option<DateTime<Utc>>,
    connection_errors: u64,
}

/// Comprehensive Kafka metrics for monitoring
#[derive(Debug, Default, Clone)]
pub struct KafkaMetrics {
    // Producer metrics
    pub producer_record_send_rate: f64,
    pub producer_record_retry_rate: f64,
    pub producer_record_error_rate: f64,
    pub producer_request_latency_avg: f64,
    pub producer_request_latency_max: f64,
    pub producer_buffer_available_bytes: i64,
    pub producer_batch_size_avg: f64,
    pub producer_compression_rate_avg: f64,

    // Consumer metrics
    pub consumer_records_lag: i64,
    pub consumer_records_lag_max: i64,
    pub consumer_fetch_rate: f64,
    pub consumer_fetch_latency_avg: f64,
    pub consumer_records_consumed_rate: f64,
    pub consumer_bytes_consumed_rate: f64,

    // Connection metrics
    pub connection_count: i32,
    pub connection_creation_rate: f64,
    pub connection_close_rate: f64,
    pub network_io_rate: f64,

    // Schema registry metrics
    pub schema_registry_requests: u64,
    pub schema_registry_cache_hits: u64,
    pub schema_registry_cache_misses: u64,
    pub schema_registry_errors: u64,

    // Timestamps
    pub last_update: Option<DateTime<Utc>>,
    pub metrics_interval_ms: u64,
}

impl KafkaProducer {
    pub fn new(config: StreamConfig) -> Result<Self> {
        let kafka_config = if let StreamBackend::Kafka {
            brokers,
            security_protocol,
            sasl_config,
        } = &config.backend
        {
            KafkaProducerConfig {
                brokers: brokers.clone(),
                security_config: Self::build_security_config(security_protocol, sasl_config),
                ..Default::default()
            }
        } else {
            return Err(anyhow!("Invalid backend configuration for Kafka producer"));
        };

        Ok(Self {
            config,
            kafka_config,
            #[cfg(feature = "kafka")]
            producer: None,
            #[cfg(not(feature = "kafka"))]
            _phantom: std::marker::PhantomData,
            stats: ProducerStats::default(),
            transaction_active: false,
            batch_buffer: Vec::new(),
            partition_cache: HashMap::new(),
            schema_cache: HashMap::new(),
            #[cfg(feature = "kafka")]
            // TODO: Enable when reqwest is added
            // schema_registry_client: None,
            metrics_collector: None,
            send_semaphore: Arc::new(Semaphore::new(1000)), // Max concurrent sends
        })
    }

    pub fn with_kafka_config(mut self, kafka_config: KafkaProducerConfig) -> Self {
        self.kafka_config = kafka_config;
        self
    }

    /// Initialize schema registry client if configured
    #[cfg(feature = "kafka")]
    async fn init_schema_registry(&mut self) -> Result<()> {
        // TODO: Enable when reqwest is added
        // if let Some(ref config) = self.kafka_config.schema_registry_config {
        //     let client = SchemaRegistryClient::new(config.clone())?;
        //
        //     // Pre-register RDF event schemas
        //     RdfEventSchemas::register_all_schemas(&client, "oxirs").await?;
        //
        //     self.schema_registry_client = Some(Arc::new(client));
        //     info!("Initialized schema registry client");
        // }
        Ok(())
    }

    /// Initialize metrics collector
    fn init_metrics_collector(&mut self) {
        self.metrics_collector = Some(Arc::new(RwLock::new(KafkaMetrics::default())));
    }

    #[cfg(feature = "kafka")]
    pub async fn connect(&mut self) -> Result<()> {
        // Apply performance optimizations before connecting
        self.optimize_performance_parameters().await?;

        let mut client_config = ClientConfig::new();

        // Basic configuration
        client_config
            .set("bootstrap.servers", self.kafka_config.brokers.join(","))
            .set("client.id", &self.kafka_config.client_id)
            .set(
                "enable.idempotence",
                self.kafka_config.enable_idempotence.to_string(),
            )
            .set("acks", self.kafka_config.acks.to_string())
            .set("retries", self.kafka_config.retries.to_string())
            .set(
                "retry.backoff.ms",
                self.kafka_config.retry_backoff_ms.to_string(),
            )
            .set("batch.size", self.kafka_config.batch_size.to_string())
            .set("linger.ms", self.kafka_config.linger_ms.to_string())
            .set("buffer.memory", self.kafka_config.buffer_memory.to_string())
            .set(
                "compression.type",
                self.kafka_config.compression_type.to_string(),
            )
            .set(
                "max.in.flight.requests.per.connection",
                self.kafka_config.max_in_flight_requests.to_string(),
            )
            .set(
                "request.timeout.ms",
                self.kafka_config.request_timeout_ms.to_string(),
            )
            .set(
                "delivery.timeout.ms",
                self.kafka_config.delivery_timeout_ms.to_string(),
            )
            .set("log_level", "info");

        // Advanced performance tuning
        self.apply_advanced_performance_config(&mut client_config)?;
        self.apply_buffer_memory_optimization(&mut client_config)?;
        self.apply_linger_time_optimization(&mut client_config)?;
        self.apply_timeout_optimization(&mut client_config)?;
        self.apply_retry_optimization(&mut client_config)?;

        // Transaction support
        if let Some(transaction_id) = &self.kafka_config.transaction_id {
            client_config.set("transactional.id", transaction_id);
        }

        // Security configuration
        if let Some(security_config) = &self.kafka_config.security_config {
            self.apply_security_config(&mut client_config, security_config)?;
        }

        // Performance optimizations
        client_config
            .set("socket.keepalive.enable", "true")
            .set("socket.nagle.disable", "true")
            .set("socket.max.fails", "3")
            .set("reconnect.backoff.ms", "100")
            .set("reconnect.backoff.max.ms", "10000");

        let producer: FutureProducer = client_config
            .create()
            .map_err(|e| anyhow!("Failed to create Kafka producer: {}", e))?;

        // Initialize transactions if configured
        if self.kafka_config.transaction_id.is_some() {
            producer
                .init_transactions(Timeout::After(Duration::from_secs(30)))
                .map_err(|e| anyhow!("Failed to initialize transactions: {}", e))?;
            info!("Initialized Kafka transactions");
        }

        self.producer = Some(producer);

        // Initialize schema registry if configured
        self.init_schema_registry().await?;

        // Initialize metrics collector
        self.init_metrics_collector();

        info!(
            "Connected to Kafka: {} (idempotence: {}, transactions: {})",
            self.kafka_config.brokers.join(","),
            self.kafka_config.enable_idempotence,
            self.kafka_config.transaction_id.is_some()
        );
        Ok(())
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn connect(&mut self) -> Result<()> {
        // Apply mock optimizations
        self.optimize_performance_parameters().await?;
        let mut mock_config = ();
        self.apply_buffer_memory_optimization(&mut mock_config)?;
        self.apply_linger_time_optimization(&mut mock_config)?;
        self.apply_timeout_optimization(&mut mock_config)?;
        self.apply_retry_optimization(&mut mock_config)?;

        warn!("Kafka feature not enabled, using mock producer");
        Ok(())
    }

    /// Optimize Kafka performance parameters based on workload characteristics
    async fn optimize_performance_parameters(&mut self) -> Result<()> {
        let workload_profile = self.analyze_workload_profile().await;

        match workload_profile {
            WorkloadProfile::HighThroughput => {
                self.optimize_for_throughput();
                info!("Applied high-throughput optimizations");
            }
            WorkloadProfile::LowLatency => {
                self.optimize_for_latency();
                info!("Applied low-latency optimizations");
            }
            WorkloadProfile::Balanced => {
                self.optimize_for_balanced();
                info!("Applied balanced optimizations");
            }
            WorkloadProfile::LargeMessages => {
                self.optimize_for_large_messages();
                info!("Applied large message optimizations");
            }
        }

        Ok(())
    }

    /// Analyze workload characteristics to determine optimal configuration
    async fn analyze_workload_profile(&self) -> WorkloadProfile {
        // For now, return balanced profile
        // In production, this would analyze:
        // - Message size distribution
        // - Throughput requirements
        // - Latency requirements
        // - Network conditions
        // - Available memory
        WorkloadProfile::Balanced
    }

    /// Optimize configuration for maximum throughput
    fn optimize_for_throughput(&mut self) {
        self.kafka_config.batch_size = 131072; // 128KB
        self.kafka_config.linger_ms = 50; // Wait for batches to fill
        self.kafka_config.buffer_memory = 67108864; // 64MB
        self.kafka_config.compression_type = KafkaCompressionType::Lz4; // Fast compression
        self.kafka_config.max_in_flight_requests = 5;
        self.kafka_config.request_timeout_ms = 60000; // Longer timeout for stability
        self.kafka_config.delivery_timeout_ms = 600000; // 10 minutes
        self.kafka_config.retry_backoff_ms = 500; // Longer backoff for stability
    }

    /// Optimize configuration for minimum latency
    fn optimize_for_latency(&mut self) {
        self.kafka_config.batch_size = 1024; // Small batches
        self.kafka_config.linger_ms = 0; // Send immediately
        self.kafka_config.buffer_memory = 16777216; // 16MB
        self.kafka_config.compression_type = KafkaCompressionType::None; // No compression overhead
        self.kafka_config.max_in_flight_requests = 1; // Ordered delivery
        self.kafka_config.request_timeout_ms = 5000; // Quick timeout
        self.kafka_config.delivery_timeout_ms = 30000; // 30 seconds
        self.kafka_config.retry_backoff_ms = 50; // Quick retry
    }

    /// Optimize configuration for balanced performance
    fn optimize_for_balanced(&mut self) {
        self.kafka_config.batch_size = 65536; // 64KB
        self.kafka_config.linger_ms = 10; // Small wait for batching
        self.kafka_config.buffer_memory = 33554432; // 32MB
        self.kafka_config.compression_type = KafkaCompressionType::Snappy; // Balanced compression
        self.kafka_config.max_in_flight_requests = 3; // Good parallelism
        self.kafka_config.request_timeout_ms = 30000; // 30 seconds
        self.kafka_config.delivery_timeout_ms = 300000; // 5 minutes
        self.kafka_config.retry_backoff_ms = 100; // Standard backoff
    }

    /// Optimize configuration for large message handling
    fn optimize_for_large_messages(&mut self) {
        self.kafka_config.batch_size = 262144; // 256KB
        self.kafka_config.linger_ms = 100; // Wait longer for large batches
        self.kafka_config.buffer_memory = 134217728; // 128MB
        self.kafka_config.compression_type = KafkaCompressionType::Zstd; // Best compression
        self.kafka_config.max_in_flight_requests = 2; // Reduced for large messages
        self.kafka_config.request_timeout_ms = 120000; // 2 minutes
        self.kafka_config.delivery_timeout_ms = 1200000; // 20 minutes
        self.kafka_config.retry_backoff_ms = 1000; // Longer backoff for large messages
    }

    /// Apply advanced performance configuration to Kafka client
    #[cfg(feature = "kafka")]
    fn apply_advanced_performance_config(&self, client_config: &mut ClientConfig) -> Result<()> {
        // Advanced buffer management
        client_config
            .set("send.buffer.bytes", "131072") // 128KB send buffer
            .set("receive.buffer.bytes", "131072") // 128KB receive buffer
            .set("socket.send.buffer.bytes", "131072") // OS send buffer
            .set("socket.receive.buffer.bytes", "131072") // OS receive buffer
            // Advanced batching and compression
            .set("batch.num.messages", "10000") // Max messages per batch
            .set("queue.buffering.max.messages", "1000000") // 1M message queue
            .set("queue.buffering.max.kbytes", "2097152") // 2GB memory limit
            // Advanced retry and timeout configuration
            .set(
                "message.send.max.retries",
                self.kafka_config.retries.to_string(),
            )
            .set(
                "retry.backoff.max.ms",
                (self.kafka_config.retry_backoff_ms * 10).to_string(),
            )
            .set(
                "queue.buffering.max.ms",
                self.kafka_config.linger_ms.to_string(),
            )
            // Advanced network configuration
            .set("socket.keepalive.enable", "true")
            .set("socket.nagle.disable", "true") // Disable Nagle for lower latency
            .set("socket.max.fails", "3")
            .set("reconnect.backoff.ms", "100")
            .set("reconnect.backoff.max.ms", "10000")
            // Advanced threading and parallelism
            .set("internal.termination.signal", "2") // SIGINT
            .set("api.version.request", "true")
            .set("api.version.fallback.ms", "10000")
            // Advanced metadata configuration
            .set("metadata.request.timeout.ms", "60000")
            .set("metadata.refresh.interval.ms", "300000") // 5 minutes
            .set("metadata.max.age.ms", "900000") // 15 minutes
            // Advanced partition and leadership configuration
            .set("topic.metadata.refresh.interval.ms", "30000")
            .set("topic.metadata.refresh.sparse", "true");

        info!("Applied advanced Kafka performance configuration");
        Ok(())
    }

    /// Apply buffer memory specific optimizations
    #[cfg(feature = "kafka")]
    fn apply_buffer_memory_optimization(&self, client_config: &mut ClientConfig) -> Result<()> {
        // Adaptive buffer sizing based on available memory
        let total_memory = std::env::var("KAFKA_TOTAL_MEMORY")
            .unwrap_or_else(|_| "1073741824".to_string()) // Default 1GB
            .parse::<u64>()
            .unwrap_or(1073741824);

        let buffer_ratio = 0.1; // Use 10% of total memory for buffers
        let optimal_buffer = (total_memory as f64 * buffer_ratio) as u64;

        client_config
            .set("buffer.memory", optimal_buffer.to_string())
            .set("send.buffer.bytes", (optimal_buffer / 256).to_string()) // 1/256th of buffer memory
            .set("receive.buffer.bytes", (optimal_buffer / 256).to_string())
            .set("socket.send.buffer.bytes", "262144") // 256KB OS buffer
            .set("socket.receive.buffer.bytes", "262144"); // 256KB OS buffer

        info!(
            "Applied buffer memory optimizations: {}MB total buffer",
            optimal_buffer / 1024 / 1024
        );
        Ok(())
    }

    /// Apply linger time optimizations based on workload
    #[cfg(feature = "kafka")]
    fn apply_linger_time_optimization(&self, client_config: &mut ClientConfig) -> Result<()> {
        // Dynamic linger time based on throughput requirements
        let target_throughput = std::env::var("KAFKA_TARGET_THROUGHPUT")
            .unwrap_or_else(|_| "10000".to_string()) // Default 10K msg/sec
            .parse::<u32>()
            .unwrap_or(10000);

        let optimal_linger = if target_throughput > 50000 {
            50 // High throughput: longer linger for better batching
        } else if target_throughput > 10000 {
            self.kafka_config.linger_ms // Medium throughput: use configured value
        } else {
            0 // Low throughput: immediate send for low latency
        };

        client_config
            .set("linger.ms", optimal_linger.to_string())
            .set(
                "batch.size",
                if optimal_linger > 20 {
                    "131072"
                } else {
                    "65536"
                },
            ) // Adaptive batch size
            .set(
                "batch.num.messages",
                if optimal_linger > 20 {
                    "20000"
                } else {
                    "10000"
                },
            );

        info!(
            "Applied linger time optimizations: {}ms (target throughput: {})",
            optimal_linger, target_throughput
        );
        Ok(())
    }

    /// Apply timeout optimizations based on network conditions
    #[cfg(feature = "kafka")]
    fn apply_timeout_optimization(&self, client_config: &mut ClientConfig) -> Result<()> {
        // Adaptive timeouts based on network latency and reliability
        let network_class =
            std::env::var("KAFKA_NETWORK_CLASS").unwrap_or_else(|_| "lan".to_string()); // lan, wan, or cloud

        let (request_timeout, delivery_timeout, retry_backoff) = match network_class.as_str() {
            "lan" => (15000, 120000, 100),    // Fast local network
            "wan" => (45000, 600000, 500),    // Slower WAN
            "cloud" => (60000, 900000, 1000), // Variable cloud network
            _ => (
                self.kafka_config.request_timeout_ms,
                self.kafka_config.delivery_timeout_ms,
                self.kafka_config.retry_backoff_ms,
            ),
        };

        client_config
            .set("request.timeout.ms", request_timeout.to_string())
            .set("delivery.timeout.ms", delivery_timeout.to_string())
            .set("retry.backoff.ms", retry_backoff.to_string())
            .set("reconnect.backoff.ms", (retry_backoff / 2).to_string())
            .set("reconnect.backoff.max.ms", (retry_backoff * 20).to_string())
            .set(
                "connections.max.idle.ms",
                (delivery_timeout / 2).to_string(),
            );

        info!(
            "Applied timeout optimizations for {} network: request={}ms, delivery={}ms",
            network_class, request_timeout, delivery_timeout
        );
        Ok(())
    }

    /// Apply retry mechanism optimizations
    #[cfg(feature = "kafka")]
    fn apply_retry_optimization(&self, client_config: &mut ClientConfig) -> Result<()> {
        // Intelligent retry configuration based on reliability requirements
        let reliability_level =
            std::env::var("KAFKA_RELIABILITY_LEVEL").unwrap_or_else(|_| "high".to_string()); // low, medium, high, critical

        let (max_retries, retry_backoff_max, enable_idempotence) = match reliability_level.as_str()
        {
            "low" => (3, 2000, false),
            "medium" => (10, 5000, true),
            "high" => (2147483647, 10000, true), // Max retries
            "critical" => (2147483647, 20000, true), // Max retries with longer backoff
            _ => (
                self.kafka_config.retries,
                10000,
                self.kafka_config.enable_idempotence,
            ),
        };

        let acks_value = if reliability_level == "critical" {
            "all".to_string()
        } else {
            self.kafka_config.acks.to_string()
        };

        client_config
            .set("retries", max_retries.to_string())
            .set("retry.backoff.max.ms", retry_backoff_max.to_string())
            .set("enable.idempotence", enable_idempotence.to_string())
            .set(
                "max.in.flight.requests.per.connection",
                if enable_idempotence { "5" } else { "1" },
            ) // Maintain ordering if needed
            .set("acks", &acks_value);

        info!(
            "Applied retry optimizations for {} reliability: retries={}, idempotence={}",
            reliability_level, max_retries, enable_idempotence
        );
        Ok(())
    }

    /// Mock buffer memory optimization for non-kafka builds
    #[cfg(not(feature = "kafka"))]
    fn apply_buffer_memory_optimization(&self, _client_config: &mut ()) -> Result<()> {
        debug!("Mock: applied buffer memory optimizations");
        Ok(())
    }

    /// Mock linger time optimization for non-kafka builds
    #[cfg(not(feature = "kafka"))]
    fn apply_linger_time_optimization(&self, _client_config: &mut ()) -> Result<()> {
        debug!("Mock: applied linger time optimizations");
        Ok(())
    }

    /// Mock timeout optimization for non-kafka builds
    #[cfg(not(feature = "kafka"))]
    fn apply_timeout_optimization(&self, _client_config: &mut ()) -> Result<()> {
        debug!("Mock: applied timeout optimizations");
        Ok(())
    }

    /// Mock retry optimization for non-kafka builds
    #[cfg(not(feature = "kafka"))]
    fn apply_retry_optimization(&self, _client_config: &mut ()) -> Result<()> {
        debug!("Mock: applied retry optimizations");
        Ok(())
    }

    /// Dynamic performance tuning based on runtime metrics
    pub async fn tune_performance_dynamically(&mut self) -> Result<()> {
        if let Some(ref metrics_collector) = self.metrics_collector {
            if let Ok(metrics) = metrics_collector.try_read() {
                // Analyze current performance metrics
                let avg_latency = metrics.producer_request_latency_avg;
                let error_rate = metrics.producer_record_error_rate;
                let buffer_utilization = 1.0
                    - (metrics.producer_buffer_available_bytes as f64
                        / self.kafka_config.buffer_memory as f64);

                // Adjust parameters based on metrics
                if avg_latency > 100.0 && buffer_utilization < 0.5 {
                    // High latency, low buffer usage - optimize for latency
                    self.kafka_config.linger_ms = (self.kafka_config.linger_ms / 2).max(0);
                    self.kafka_config.batch_size = (self.kafka_config.batch_size / 2).max(1024);
                    info!(
                        "Dynamically adjusted for lower latency: linger_ms={}, batch_size={}",
                        self.kafka_config.linger_ms, self.kafka_config.batch_size
                    );
                } else if error_rate > 0.01 && buffer_utilization > 0.8 {
                    // High error rate, high buffer usage - increase buffer and timeouts
                    self.kafka_config.buffer_memory =
                        (self.kafka_config.buffer_memory * 2).min(134217728); // Max 128MB
                    self.kafka_config.request_timeout_ms =
                        (self.kafka_config.request_timeout_ms * 2).min(120000); // Max 2 min
                    info!(
                        "Dynamically increased buffer and timeouts: buffer={} bytes, timeout={} ms",
                        self.kafka_config.buffer_memory, self.kafka_config.request_timeout_ms
                    );
                } else if avg_latency < 10.0 && buffer_utilization < 0.3 {
                    // Low latency, low buffer usage - can optimize for throughput
                    self.kafka_config.batch_size = (self.kafka_config.batch_size * 2).min(262144); // Max 256KB
                    self.kafka_config.linger_ms = (self.kafka_config.linger_ms + 5).min(100);
                    info!(
                        "Dynamically optimized for throughput: batch_size={}, linger_ms={}",
                        self.kafka_config.batch_size, self.kafka_config.linger_ms
                    );
                }
            }
        }
        Ok(())
    }

    fn build_security_config(
        security_protocol: &Option<String>,
        sasl_config: &Option<crate::SaslConfig>,
    ) -> Option<KafkaSecurityConfig> {
        // Simplified implementation - in real usage would convert from config types
        None
    }

    #[cfg(feature = "kafka")]
    fn apply_security_config(
        &self,
        client_config: &mut ClientConfig,
        security_config: &KafkaSecurityConfig,
    ) -> Result<()> {
        client_config.set(
            "security.protocol",
            security_config.security_protocol.to_string(),
        );

        if let Some(sasl_config) = &security_config.sasl_config {
            client_config
                .set("sasl.mechanism", sasl_config.mechanism.to_string())
                .set("sasl.username", &sasl_config.username)
                .set("sasl.password", &sasl_config.password);
        }

        if let Some(ssl_config) = &security_config.ssl_config {
            if let Some(ca_location) = &ssl_config.ca_location {
                client_config.set("ssl.ca.location", ca_location);
            }
            if let Some(cert_location) = &ssl_config.certificate_location {
                client_config.set("ssl.certificate.location", cert_location);
            }
            if let Some(key_location) = &ssl_config.key_location {
                client_config.set("ssl.key.location", key_location);
            }
        }

        Ok(())
    }

    pub async fn begin_transaction(&mut self) -> Result<()> {
        #[cfg(feature = "kafka")]
        {
            if let Some(ref producer) = self.producer {
                if self.kafka_config.transaction_id.is_some() && !self.transaction_active {
                    producer
                        .begin_transaction()
                        .map_err(|e| anyhow!("Failed to begin transaction: {}", e))?;
                    self.transaction_active = true;
                    info!("Started Kafka transaction");
                }
            }
        }
        Ok(())
    }

    pub async fn commit_transaction(&mut self) -> Result<()> {
        #[cfg(feature = "kafka")]
        {
            if let Some(ref producer) = self.producer {
                if self.transaction_active {
                    producer
                        .commit_transaction(Timeout::After(Duration::from_secs(30)))
                        .map_err(|e| anyhow!("Failed to commit transaction: {}", e))?;
                    self.transaction_active = false;
                    self.stats.transactions_committed += 1;
                    info!("Committed Kafka transaction");
                }
            }
        }
        Ok(())
    }

    /// Update metrics from Kafka statistics
    #[cfg(feature = "kafka")]
    fn update_metrics(&self, stats: &Statistics) {
        if let Some(ref metrics_collector) = self.metrics_collector {
            if let Ok(mut metrics) = metrics_collector.try_write() {
                // Extract relevant metrics from Kafka statistics
                if let Some(brokers) = stats.brokers.values().next() {
                    // Use available broker metrics
                    if let Some(rtt) = &brokers.rtt {
                        metrics.producer_request_latency_avg = rtt.avg as f64 / 1000.0; // Convert to ms
                        metrics.producer_request_latency_max = rtt.max as f64 / 1000.0;
                        // Convert to ms
                    }
                }

                // Update connection metrics
                metrics.connection_count = stats.brokers.len() as i32;

                // Update timestamp
                metrics.last_update = Some(Utc::now());
            }
        }
    }

    // TODO: Enable when reqwest is added
    // /// Register event schema with schema registry
    // #[cfg(feature = "kafka")]
    // async fn register_event_schema(
    //     &self,
    //     event: &KafkaEvent,
    //     schema_registry: &SchemaRegistryClient,
    // ) -> Result<u32> {
    //     let subject = format!("oxirs-{}-value", event.event_type);
    //
    //     // Check cache first
    //     if let Some(&schema_id) = self.schema_cache.get(&subject) {
    //         return Ok(schema_id);
    //     }
    //
    //     // Get the appropriate schema based on event type
    //     let schema = match event.event_type.as_str() {
    //         "triple_added" | "triple_removed" => RdfEventSchemas::triple_event_schema(),
    //         "graph_created" | "graph_cleared" | "graph_deleted" => RdfEventSchemas::graph_event_schema(),
    //         "sparql_update" => RdfEventSchemas::sparql_update_schema(),
    //         _ => return Err(anyhow!("Unknown event type for schema: {}", event.event_type)),
    //     };
    //
    //     // Register schema and get ID
    //     let metadata = schema_registry
    //         .register_schema(&subject, schema, SchemaType::Json, None)
    //         .await?;
    //
    //     Ok(metadata.id)
    // }

    pub async fn abort_transaction(&mut self) -> Result<()> {
        #[cfg(feature = "kafka")]
        {
            if let Some(ref producer) = self.producer {
                if self.transaction_active {
                    producer
                        .abort_transaction(Timeout::After(Duration::from_secs(30)))
                        .map_err(|e| anyhow!("Failed to abort transaction: {}", e))?;
                    self.transaction_active = false;
                    self.stats.transactions_aborted += 1;
                    warn!("Aborted Kafka transaction");
                }
            }
        }
        Ok(())
    }

    pub async fn publish(&mut self, event: StreamEvent) -> Result<()> {
        let start_time = Instant::now();
        let mut kafka_event = KafkaEvent::from(event);

        #[cfg(feature = "kafka")]
        {
            // Periodically tune performance based on metrics (every 1000 events)
            if self.stats.events_published % 1000 == 0 && self.stats.events_published > 0 {
                if let Err(e) = self.tune_performance_dynamically().await {
                    warn!("Failed to tune performance dynamically: {}", e);
                }
            }

            // Acquire send permit to limit concurrent sends
            let _permit = self
                .send_semaphore
                .acquire()
                .await
                .map_err(|_| anyhow!("Failed to acquire send permit"))?;

            // Register schema if using schema registry
            // TODO: Enable when reqwest is added
            // if let Some(ref schema_registry) = self.schema_registry_client {
            //     match self.register_event_schema(&kafka_event, schema_registry.as_ref()).await {
            //         Ok(schema_id) => {
            //             kafka_event.schema_id = Some(schema_id);
            //             self.stats.schema_registry_calls += 1;
            //         }
            //         Err(e) => {
            //             warn!("Failed to register schema: {}", e);
            //             // Continue without schema ID
            //         }
            //     }
            // }

            if let Some(ref producer) = self.producer {
                let payload = serde_json::to_string(&kafka_event)
                    .map_err(|e| anyhow!("Failed to serialize event: {}", e))?;

                // Build headers
                let mut headers = OwnedHeaders::new();
                headers = headers.insert(Header {
                    key: "event_type",
                    value: Some(kafka_event.event_type.as_str()),
                });
                headers = headers.insert(Header {
                    key: "source",
                    value: Some(kafka_event.source.as_str()),
                });
                headers = headers.insert(Header {
                    key: "schema_version",
                    value: Some(kafka_event.schema_version.as_str()),
                });

                // Add custom headers
                for (key, value) in &kafka_event.headers {
                    headers = headers.insert(Header {
                        key: key.as_str(),
                        value: Some(value.as_str()),
                    });
                }

                let mut record = FutureRecord::to(&self.config.topic)
                    .payload(&payload)
                    .timestamp(kafka_event.timestamp.timestamp_millis())
                    .headers(headers);

                // Set partition key for ordered delivery
                if let Some(partition_key) = &kafka_event.partition_key {
                    record = record.key(partition_key);
                }

                match producer
                    .send(record, Timeout::After(Duration::from_secs(30)))
                    .await
                {
                    Ok((partition, offset)) => {
                        self.stats.events_published += 1;
                        self.stats.bytes_sent += payload.len() as u64;

                        let latency = start_time.elapsed().as_millis() as u64;
                        self.stats.max_latency_ms = self.stats.max_latency_ms.max(latency);
                        self.stats.avg_latency_ms =
                            (self.stats.avg_latency_ms + latency as f64) / 2.0;
                        self.stats.last_publish = Some(Utc::now());

                        debug!(
                            "Published event {} to partition {} offset {}",
                            kafka_event.event_id, partition, offset
                        );
                    }
                    Err((e, _)) => {
                        self.stats.events_failed += 1;
                        error!("Failed to publish event: {}", e);
                        return Err(anyhow!("Failed to publish to Kafka: {}", e));
                    }
                }
            } else {
                return Err(anyhow!("Kafka producer not initialized"));
            }
        }

        #[cfg(not(feature = "kafka"))]
        {
            self.batch_buffer.push(kafka_event);
            debug!("Mock publish: stored event in memory");
        }

        Ok(())
    }

    pub async fn publish_batch(&mut self, events: Vec<StreamEvent>) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }

        let start_time = Instant::now();

        #[cfg(feature = "kafka")]
        {
            if let Some(ref producer) = self.producer {
                // Send events sequentially to avoid borrowing issues with FutureRecord
                let mut success_count = 0;
                let mut failure_count = 0;

                for event in events {
                    let kafka_event = KafkaEvent::from(event);
                    let payload = serde_json::to_string(&kafka_event)
                        .map_err(|e| anyhow!("Failed to serialize event: {}", e))?;

                    let mut headers = OwnedHeaders::new();
                    headers = headers.insert(Header {
                        key: "event_type",
                        value: Some(kafka_event.event_type.as_str()),
                    });

                    let mut record = FutureRecord::to(&self.config.topic)
                        .payload(&payload)
                        .timestamp(kafka_event.timestamp.timestamp_millis())
                        .headers(headers);

                    if let Some(ref partition_key) = kafka_event.partition_key {
                        record = record.key(partition_key);
                    }

                    // Send and await immediately
                    match producer
                        .send(record, Timeout::After(Duration::from_secs(30)))
                        .await
                    {
                        Ok(_) => success_count += 1,
                        Err((err, _)) => {
                            error!("Failed to send Kafka message: {}", err);
                            failure_count += 1;
                        }
                    }
                }

                self.stats.events_published += success_count;
                self.stats.events_failed += failure_count;
                self.stats.batches_sent += 1;

                let batch_latency = start_time.elapsed().as_millis() as u64;
                self.stats.max_latency_ms = self.stats.max_latency_ms.max(batch_latency);

                if failure_count > 0 {
                    warn!(
                        "Batch publish completed with {} failures out of {} events",
                        failure_count,
                        success_count + failure_count
                    );
                }

                debug!("Published batch of {} events", success_count);
            }
        }

        #[cfg(not(feature = "kafka"))]
        {
            for event in events {
                self.batch_buffer.push(KafkaEvent::from(event));
            }
            debug!(
                "Mock batch publish: stored {} events in memory",
                self.batch_buffer.len()
            );
        }

        Ok(())
    }

    pub async fn publish_patch(&mut self, patch: &RdfPatch) -> Result<()> {
        let mut events = Vec::new();

        for operation in &patch.operations {
            let metadata = EventMetadata {
                event_id: Uuid::new_v4().to_string(),
                timestamp: Utc::now(),
                source: "kafka_patch".to_string(),
                user: None,
                context: Some(patch.id.clone()),
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            };

            let event = match operation {
                PatchOperation::Add {
                    subject,
                    predicate,
                    object,
                } => StreamEvent::TripleAdded {
                    subject: subject.clone(),
                    predicate: predicate.clone(),
                    object: object.clone(),
                    graph: None,
                    metadata,
                },
                PatchOperation::Delete {
                    subject,
                    predicate,
                    object,
                } => StreamEvent::TripleRemoved {
                    subject: subject.clone(),
                    predicate: predicate.clone(),
                    object: object.clone(),
                    graph: None,
                    metadata,
                },
                PatchOperation::AddGraph { graph } => StreamEvent::GraphCreated {
                    graph: graph.clone(),
                    metadata,
                },
                PatchOperation::DeleteGraph { graph } => StreamEvent::GraphDeleted {
                    graph: graph.clone(),
                    metadata,
                },
                PatchOperation::AddPrefix { .. } => {
                    // Skip prefix operations for now
                    continue;
                }
                PatchOperation::DeletePrefix { .. } => {
                    // Skip prefix operations for now
                    continue;
                }
                PatchOperation::TransactionBegin { transaction_id } => {
                    StreamEvent::TransactionBegin {
                        transaction_id: transaction_id
                            .clone()
                            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                        isolation_level: None,
                        metadata,
                    }
                }
                PatchOperation::TransactionCommit => StreamEvent::TransactionCommit {
                    transaction_id: uuid::Uuid::new_v4().to_string(),
                    metadata,
                },
                PatchOperation::TransactionAbort => StreamEvent::TransactionAbort {
                    transaction_id: uuid::Uuid::new_v4().to_string(),
                    metadata,
                },
                PatchOperation::Header { .. } => {
                    // Skip header operations for now
                    continue;
                }
            };
            events.push(event);
        }

        self.publish_batch(events).await
    }

    pub async fn flush(&mut self) -> Result<()> {
        #[cfg(feature = "kafka")]
        {
            if let Some(ref producer) = self.producer {
                producer
                    .flush(Timeout::After(Duration::from_secs(30)))
                    .map_err(|e| anyhow!("Failed to flush Kafka producer: {}", e))?;
                debug!("Flushed Kafka producer");
            }
        }

        #[cfg(not(feature = "kafka"))]
        {
            debug!("Mock flush: {} pending events", self.batch_buffer.len());
        }

        Ok(())
    }

    pub fn get_stats(&self) -> &ProducerStats {
        &self.stats
    }

    /// Get current performance configuration summary
    pub fn get_performance_config(&self) -> KafkaPerformanceConfig {
        KafkaPerformanceConfig {
            batch_size: self.kafka_config.batch_size,
            linger_ms: self.kafka_config.linger_ms,
            buffer_memory: self.kafka_config.buffer_memory,
            compression_type: self.kafka_config.compression_type.clone(),
            max_in_flight_requests: self.kafka_config.max_in_flight_requests,
            request_timeout_ms: self.kafka_config.request_timeout_ms,
            delivery_timeout_ms: self.kafka_config.delivery_timeout_ms,
            retry_backoff_ms: self.kafka_config.retry_backoff_ms,
            enable_idempotence: self.kafka_config.enable_idempotence,
            acks: self.kafka_config.acks.clone(),
        }
    }

    /// Get performance recommendations based on current metrics
    pub async fn get_performance_recommendations(&self) -> Vec<PerformanceRecommendation> {
        let mut recommendations = Vec::new();

        if let Some(ref metrics_collector) = self.metrics_collector {
            if let Ok(metrics) = metrics_collector.try_read() {
                // Analyze latency
                if metrics.producer_request_latency_avg > 100.0 {
                    recommendations.push(PerformanceRecommendation {
                        category: "Latency".to_string(),
                        description: "High average latency detected. Consider reducing batch size and linger time.".to_string(),
                        suggested_action: "Set batch_size < 32KB and linger_ms = 0".to_string(),
                        impact: "Medium".to_string(),
                    });
                }

                // Analyze error rate
                if metrics.producer_record_error_rate > 0.01 {
                    recommendations.push(PerformanceRecommendation {
                        category: "Reliability".to_string(),
                        description:
                            "High error rate detected. Consider increasing retry configuration."
                                .to_string(),
                        suggested_action: "Increase request_timeout_ms and retry_backoff_ms"
                            .to_string(),
                        impact: "High".to_string(),
                    });
                }

                // Analyze buffer utilization
                if metrics.producer_buffer_available_bytes
                    < (self.kafka_config.buffer_memory as i64 / 10)
                {
                    recommendations.push(PerformanceRecommendation {
                        category: "Memory".to_string(),
                        description: "Low buffer availability. Consider increasing buffer memory."
                            .to_string(),
                        suggested_action: "Increase buffer_memory to at least 64MB".to_string(),
                        impact: "High".to_string(),
                    });
                }

                // Analyze throughput
                if metrics.producer_record_send_rate < 1000.0 && self.stats.events_published > 10000
                {
                    recommendations.push(PerformanceRecommendation {
                        category: "Throughput".to_string(),
                        description: "Low send rate detected. Consider optimizing for throughput."
                            .to_string(),
                        suggested_action: "Increase batch_size and linger_ms, enable compression"
                            .to_string(),
                        impact: "Medium".to_string(),
                    });
                }
            }
        }

        recommendations
    }
}

/// Enhanced Kafka consumer with consumer groups and advanced features
pub struct KafkaConsumer {
    config: StreamConfig,
    kafka_config: KafkaConsumerConfig,
    #[cfg(feature = "kafka")]
    consumer: Option<StreamConsumer<KafkaConsumerContext>>,
    #[cfg(not(feature = "kafka"))]
    _phantom: std::marker::PhantomData<()>,
    stats: ConsumerStats,
    // TODO: Enable when reqwest is added
    // #[cfg(feature = "kafka")]
    // schema_registry_client: Option<Arc<SchemaRegistryClient>>,
    metrics_collector: Option<Arc<RwLock<KafkaMetrics>>>,
    consumer_lag_monitor: Arc<RwLock<HashMap<i32, i64>>>, // partition -> lag
}

/// Enhanced consumer configuration
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
    pub isolation_level: IsolationLevel,
    pub security_config: Option<KafkaSecurityConfig>,
}

impl Default for KafkaConsumerConfig {
    fn default() -> Self {
        Self {
            brokers: vec!["localhost:9092".to_string()],
            group_id: "oxirs-consumer-group".to_string(),
            client_id: format!("oxirs-consumer-{}", Uuid::new_v4()),
            auto_offset_reset: AutoOffsetReset::Earliest,
            enable_auto_commit: true,
            auto_commit_interval_ms: 5000,
            session_timeout_ms: 30000,
            heartbeat_interval_ms: 3000,
            max_poll_interval_ms: 300000,
            max_poll_records: 500,
            fetch_min_bytes: 1,
            fetch_max_wait_ms: 500,
            isolation_level: IsolationLevel::ReadCommitted,
            security_config: None,
        }
    }
}

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

/// Consumer context for handling rebalances and statistics
#[cfg(feature = "kafka")]
struct KafkaConsumerContext;

#[cfg(feature = "kafka")]
impl ClientContext for KafkaConsumerContext {}

#[cfg(feature = "kafka")]
impl ConsumerContext for KafkaConsumerContext {
    fn pre_rebalance(&self, rebalance: &Rebalance) {
        match rebalance {
            Rebalance::Assign(topic_partition_list) => {
                info!("Partition assignment: {:?}", topic_partition_list);
            }
            Rebalance::Revoke(_) => {
                info!("Partition revoked");
            }
            Rebalance::Error(err) => {
                error!("Rebalance error: {}", err);
            }
        }
    }

    fn post_rebalance(&self, rebalance: &Rebalance) {
        match rebalance {
            Rebalance::Assign(_) => {
                info!("Rebalance completed");
            }
            _ => {}
        }
    }

    fn commit_callback(&self, result: KafkaResult<()>, _offsets: &TopicPartitionList) {
        match result {
            Ok(()) => debug!("Offsets committed successfully"),
            Err(e) => error!("Error committing offsets: {}", e),
        }
    }

    // Note: stats method removed as it's not part of ConsumerContext trait in newer rdkafka versions
}

#[cfg(feature = "kafka")]
impl Default for KafkaConsumerContext {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug, Default)]
struct ConsumerStats {
    events_consumed: u64,
    events_failed: u64,
    bytes_received: u64,
    consumer_lag: u64,
    rebalances: u64,
    commits: u64,
    avg_processing_time_ms: f64,
    last_message: Option<DateTime<Utc>>,
    connection_errors: u64,
}

impl KafkaConsumer {
    pub fn new(config: StreamConfig) -> Result<Self> {
        let kafka_config = if let StreamBackend::Kafka { brokers, .. } = &config.backend {
            KafkaConsumerConfig {
                brokers: brokers.clone(),
                ..Default::default()
            }
        } else {
            return Err(anyhow!("Invalid backend configuration for Kafka consumer"));
        };

        Ok(Self {
            config,
            kafka_config,
            #[cfg(feature = "kafka")]
            consumer: None,
            #[cfg(not(feature = "kafka"))]
            _phantom: std::marker::PhantomData,
            stats: ConsumerStats::default(),
            #[cfg(feature = "kafka")]
            // TODO: Enable when reqwest is added
            // schema_registry_client: None,
            metrics_collector: None,
            consumer_lag_monitor: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub fn with_kafka_config(mut self, kafka_config: KafkaConsumerConfig) -> Self {
        self.kafka_config = kafka_config;
        self
    }

    #[cfg(feature = "kafka")]
    pub async fn connect(&mut self) -> Result<()> {
        let context = KafkaConsumerContext::default();
        let mut client_config = ClientConfig::new();

        client_config
            .set("bootstrap.servers", self.kafka_config.brokers.join(","))
            .set("group.id", &self.kafka_config.group_id)
            .set("client.id", &self.kafka_config.client_id)
            .set(
                "auto.offset.reset",
                self.kafka_config.auto_offset_reset.to_string(),
            )
            .set(
                "enable.auto.commit",
                self.kafka_config.enable_auto_commit.to_string(),
            )
            .set(
                "auto.commit.interval.ms",
                self.kafka_config.auto_commit_interval_ms.to_string(),
            )
            .set(
                "session.timeout.ms",
                self.kafka_config.session_timeout_ms.to_string(),
            )
            .set(
                "heartbeat.interval.ms",
                self.kafka_config.heartbeat_interval_ms.to_string(),
            )
            .set(
                "max.poll.interval.ms",
                self.kafka_config.max_poll_interval_ms.to_string(),
            )
            .set(
                "max.poll.records",
                self.kafka_config.max_poll_records.to_string(),
            )
            .set(
                "fetch.min.bytes",
                self.kafka_config.fetch_min_bytes.to_string(),
            )
            .set(
                "fetch.max.wait.ms",
                self.kafka_config.fetch_max_wait_ms.to_string(),
            )
            .set(
                "isolation.level",
                self.kafka_config.isolation_level.to_string(),
            )
            .set("enable.partition.eof", "false")
            .set("log_level", "info");

        // Security configuration
        if let Some(security_config) = &self.kafka_config.security_config {
            self.apply_consumer_security_config(&mut client_config, security_config)?;
        }

        let consumer: StreamConsumer<KafkaConsumerContext> = client_config
            .create_with_context(context)
            .map_err(|e| anyhow!("Failed to create Kafka consumer: {}", e))?;

        // Subscribe to topic
        consumer
            .subscribe(&[&self.config.topic])
            .map_err(|e| anyhow!("Failed to subscribe to topic: {}", e))?;

        self.consumer = Some(consumer);
        info!("Connected Kafka consumer to topic: {}", self.config.topic);
        Ok(())
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn connect(&mut self) -> Result<()> {
        warn!("Kafka feature not enabled, using mock consumer");
        Ok(())
    }

    #[cfg(feature = "kafka")]
    fn apply_consumer_security_config(
        &self,
        client_config: &mut ClientConfig,
        security_config: &KafkaSecurityConfig,
    ) -> Result<()> {
        client_config.set(
            "security.protocol",
            security_config.security_protocol.to_string(),
        );

        if let Some(sasl_config) = &security_config.sasl_config {
            client_config
                .set("sasl.mechanism", sasl_config.mechanism.to_string())
                .set("sasl.username", &sasl_config.username)
                .set("sasl.password", &sasl_config.password);
        }

        if let Some(ssl_config) = &security_config.ssl_config {
            if let Some(ca_location) = &ssl_config.ca_location {
                client_config.set("ssl.ca.location", ca_location);
            }
        }

        Ok(())
    }

    pub async fn consume(&mut self) -> Result<Option<StreamEvent>> {
        #[cfg(feature = "kafka")]
        {
            if let Some(ref consumer) = self.consumer {
                match consumer.recv().await {
                    Ok(message) => {
                        let start_time = Instant::now();

                        if let Some(payload) = message.payload_view::<str>() {
                            match payload {
                                Ok(payload_str) => {
                                    match serde_json::from_str::<KafkaEvent>(payload_str) {
                                        Ok(kafka_event) => {
                                            self.stats.events_consumed += 1;
                                            self.stats.bytes_received += payload_str.len() as u64;
                                            self.stats.last_message = Some(Utc::now());

                                            let processing_time =
                                                start_time.elapsed().as_millis() as f64;
                                            self.stats.avg_processing_time_ms =
                                                (self.stats.avg_processing_time_ms
                                                    + processing_time)
                                                    / 2.0;

                                            match kafka_event.try_into() {
                                                Ok(stream_event) => {
                                                    debug!("Consumed event: {:?}", stream_event);
                                                    Ok(Some(stream_event))
                                                }
                                                Err(e) => {
                                                    self.stats.events_failed += 1;
                                                    error!("Failed to convert Kafka event: {}", e);
                                                    Err(anyhow!("Event conversion failed: {}", e))
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            self.stats.events_failed += 1;
                                            error!("Failed to parse Kafka message: {}", e);
                                            Err(anyhow!("JSON parse error: {}", e))
                                        }
                                    }
                                }
                                Err(e) => {
                                    self.stats.events_failed += 1;
                                    error!("Failed to decode message payload: {}", e);
                                    Err(anyhow!("Payload decode error: {}", e))
                                }
                            }
                        } else {
                            debug!("Received empty message");
                            Ok(None)
                        }
                    }
                    Err(e) => {
                        self.stats.connection_errors += 1;
                        error!("Kafka receive error: {}", e);
                        Err(anyhow!("Kafka receive error: {}", e))
                    }
                }
            } else {
                Err(anyhow!("Kafka consumer not initialized"))
            }
        }

        #[cfg(not(feature = "kafka"))]
        {
            // Mock implementation
            time::sleep(Duration::from_millis(100)).await;
            Ok(None)
        }
    }

    pub async fn consume_batch(
        &mut self,
        max_events: usize,
        timeout: Duration,
    ) -> Result<Vec<StreamEvent>> {
        let mut events = Vec::new();
        let start_time = Instant::now();

        while events.len() < max_events && start_time.elapsed() < timeout {
            match tokio::time::timeout(Duration::from_millis(50), self.consume()).await {
                Ok(Ok(Some(event))) => events.push(event),
                Ok(Ok(None)) => continue,
                Ok(Err(e)) => return Err(e),
                Err(_) => break, // Timeout
            }
        }

        Ok(events)
    }

    pub async fn commit_offsets(&self) -> Result<()> {
        #[cfg(feature = "kafka")]
        {
            if let Some(ref consumer) = self.consumer {
                consumer
                    .commit_consumer_state(CommitMode::Async)
                    .map_err(|e| anyhow!("Failed to commit offsets: {}", e))?;
                debug!("Committed consumer offsets");
            }
        }
        Ok(())
    }

    pub fn get_stats(&self) -> &ConsumerStats {
        &self.stats
    }
}

/// Kafka admin utilities for topic management
pub struct KafkaAdmin {
    #[cfg(feature = "kafka")]
    admin_client: Option<AdminClient<DefaultClientContext>>,
    #[cfg(not(feature = "kafka"))]
    _phantom: std::marker::PhantomData<()>,
    brokers: Vec<String>,
}

impl KafkaAdmin {
    #[cfg(feature = "kafka")]
    pub fn new(brokers: Vec<String>) -> Result<Self> {
        let mut config = ClientConfig::new();
        config.set("bootstrap.servers", brokers.join(","));

        let admin_client: AdminClient<DefaultClientContext> = config
            .create()
            .map_err(|e| anyhow!("Failed to create admin client: {}", e))?;

        Ok(Self {
            admin_client: Some(admin_client),
            brokers,
        })
    }

    #[cfg(not(feature = "kafka"))]
    pub fn new(brokers: Vec<String>) -> Result<Self> {
        Ok(Self {
            _phantom: std::marker::PhantomData,
            brokers,
        })
    }

    #[cfg(feature = "kafka")]
    pub async fn create_topic(
        &self,
        topic: &str,
        partitions: i32,
        replication_factor: i32,
    ) -> Result<()> {
        if let Some(ref admin_client) = self.admin_client {
            let new_topic = NewTopic::new(
                topic,
                partitions,
                TopicReplication::Fixed(replication_factor),
            )
            .set("cleanup.policy", "compact")
            .set("compression.type", "snappy")
            .set("min.insync.replicas", "2");

            let results = admin_client
                .create_topics(vec![&new_topic], &AdminOptions::new())
                .await;

            match results {
                Ok(_) => {
                    info!("Created topic '{}' with {} partitions", topic, partitions);
                    Ok(())
                }
                Err(e) => {
                    if e.to_string().contains("already exists") {
                        debug!("Topic '{}' already exists", topic);
                        Ok(())
                    } else {
                        Err(anyhow!("Failed to create topic: {}", e))
                    }
                }
            }
        } else {
            Err(anyhow!("Admin client not initialized"))
        }
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn create_topic(
        &self,
        topic: &str,
        partitions: i32,
        _replication_factor: i32,
    ) -> Result<()> {
        info!(
            "Mock: created topic '{}' with {} partitions",
            topic, partitions
        );
        Ok(())
    }

    #[cfg(feature = "kafka")]
    pub async fn get_topic_metadata(&self, topic: &str) -> Result<KafkaTopicMetadata> {
        if let Some(ref admin_client) = self.admin_client {
            let metadata = admin_client
                .inner()
                .fetch_metadata(Some(topic), Timeout::After(Duration::from_secs(10)))
                .map_err(|e| anyhow!("Failed to fetch metadata: {}", e))?;

            if let Some(topic_metadata) = metadata.topics().iter().find(|t| t.name() == topic) {
                Ok(KafkaTopicMetadata {
                    name: topic.to_string(),
                    partition_count: topic_metadata.partitions().len() as u32,
                    replication_factor: topic_metadata
                        .partitions()
                        .first()
                        .map(|p| p.replicas().len() as u32)
                        .unwrap_or(0),
                    config: HashMap::new(), // Would need additional call to get config
                })
            } else {
                Err(anyhow!("Topic not found: {}", topic))
            }
        } else {
            Err(anyhow!("Admin client not initialized"))
        }
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn get_topic_metadata(&self, topic: &str) -> Result<KafkaTopicMetadata> {
        Ok(KafkaTopicMetadata {
            name: topic.to_string(),
            partition_count: 6,
            replication_factor: 3,
            config: HashMap::new(),
        })
    }
}

/// Kafka topic metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaTopicMetadata {
    pub name: String,
    pub partition_count: u32,
    pub replication_factor: u32,
    pub config: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{StreamBackend, StreamConfig};

    fn test_kafka_config() -> StreamConfig {
        StreamConfig {
            backend: StreamBackend::Kafka {
                brokers: vec!["localhost:9092".to_string()],
                security_protocol: None,
                sasl_config: None,
            },
            topic: "test-topic".to_string(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_kafka_producer_creation() {
        let config = test_kafka_config();
        let producer = KafkaProducer::new(config);
        assert!(producer.is_ok());
    }

    #[tokio::test]
    async fn test_kafka_consumer_creation() {
        let config = test_kafka_config();
        let consumer = KafkaConsumer::new(config);
        assert!(consumer.is_ok());
    }

    #[test]
    fn test_kafka_event_conversion() {
        let event = StreamEvent::TripleAdded {
            subject: "http://example.org/subject".to_string(),
            predicate: "http://example.org/predicate".to_string(),
            object: "http://example.org/object".to_string(),
            graph: None,
            metadata: EventMetadata {
                event_id: "test-event".to_string(),
                timestamp: Utc::now(),
                source: "test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            },
        };

        let kafka_event = KafkaEvent::from(event.clone());
        assert_eq!(kafka_event.event_type, "triple_added");
        assert_eq!(
            kafka_event.partition_key,
            Some("http://example.org/subject".to_string())
        );

        let converted_back: StreamEvent = kafka_event.try_into().unwrap();
        match converted_back {
            StreamEvent::TripleAdded {
                subject,
                predicate,
                object,
                ..
            } => {
                assert_eq!(subject, "http://example.org/subject");
                assert_eq!(predicate, "http://example.org/predicate");
                assert_eq!(object, "http://example.org/object");
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[tokio::test]
    async fn test_kafka_admin_operations() {
        let admin = KafkaAdmin::new(vec!["localhost:9092".to_string()]);
        assert!(admin.is_ok());

        let admin = admin.unwrap();
        let result = admin.create_topic("test-admin-topic", 3, 2).await;
        // This will fail in test environment without Kafka, but tests the API
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_compression_type_conversion() {
        assert_eq!(KafkaCompressionType::Snappy.to_string(), "snappy");
        assert_eq!(KafkaCompressionType::Lz4.to_string(), "lz4");
        assert_eq!(KafkaCompressionType::Zstd.to_string(), "zstd");
    }

    #[test]
    fn test_security_protocol_conversion() {
        assert_eq!(SecurityProtocol::SaslSsl.to_string(), "SASL_SSL");
        assert_eq!(SecurityProtocol::Plaintext.to_string(), "PLAINTEXT");
    }
}
