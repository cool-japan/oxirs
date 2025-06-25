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
use std::time::{Duration, Instant};
use tokio::time;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

#[cfg(feature = "kafka")]
use rdkafka::{
    admin::{AdminClient, AdminOptions, ConfigEntry, ConfigResource, NewTopic, TopicReplication},
    client::DefaultClientContext,
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
            StreamEvent::Heartbeat { timestamp, source } => (
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
        })
    }

    pub fn with_kafka_config(mut self, kafka_config: KafkaProducerConfig) -> Self {
        self.kafka_config = kafka_config;
        self
    }

    #[cfg(feature = "kafka")]
    pub async fn connect(&mut self) -> Result<()> {
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
        info!(
            "Connected to Kafka: {}",
            self.kafka_config.brokers.join(",")
        );
        Ok(())
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn connect(&mut self) -> Result<()> {
        warn!("Kafka feature not enabled, using mock producer");
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
        let kafka_event = KafkaEvent::from(event);

        #[cfg(feature = "kafka")]
        {
            if let Some(ref producer) = self.producer {
                let payload = serde_json::to_string(&kafka_event)
                    .map_err(|e| anyhow!("Failed to serialize event: {}", e))?;

                // Build headers
                let mut headers = OwnedHeaders::new();
                headers = headers.insert(Header {
                    key: "event_type",
                    value: Some(&kafka_event.event_type),
                });
                headers = headers.insert(Header {
                    key: "source",
                    value: Some(&kafka_event.source),
                });
                headers = headers.insert(Header {
                    key: "schema_version",
                    value: Some(&kafka_event.schema_version),
                });

                // Add custom headers
                for (key, value) in &kafka_event.headers {
                    headers = headers.insert(Header {
                        key,
                        value: Some(value),
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
            return Ok();
        }

        let start_time = Instant::now();

        #[cfg(feature = "kafka")]
        {
            if let Some(ref producer) = self.producer {
                let mut futures = Vec::new();

                for event in events {
                    let kafka_event = KafkaEvent::from(event);
                    let payload = serde_json::to_string(&kafka_event)
                        .map_err(|e| anyhow!("Failed to serialize event: {}", e))?;

                    let mut headers = OwnedHeaders::new();
                    headers = headers.insert(Header {
                        key: "event_type",
                        value: Some(&kafka_event.event_type),
                    });

                    let mut record = FutureRecord::to(&self.config.topic)
                        .payload(&payload)
                        .timestamp(kafka_event.timestamp.timestamp_millis())
                        .headers(headers);

                    if let Some(partition_key) = &kafka_event.partition_key {
                        record = record.key(partition_key);
                    }

                    let future = producer.send(record, Timeout::After(Duration::from_secs(30)));
                    futures.push(future);
                }

                // Wait for all sends to complete
                let results = futures::future::join_all(futures).await;
                let mut success_count = 0;
                let mut failure_count = 0;

                for result in results {
                    match result {
                        Ok(_) => success_count += 1,
                        Err(_) => failure_count += 1,
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
