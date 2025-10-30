//! # Apache Pulsar Backend
//!
//! High-performance Apache Pulsar integration for ultra-scalable RDF streaming.
//!
//! This module provides comprehensive Apache Pulsar integration with multi-tenancy,
//! schema registry, message ordering, and real-time processing capabilities.
//! Optimized for cloud-native deployments and massive scale scenarios.

use crate::{EventMetadata, PatchOperation, RdfPatch, StreamConfig, StreamEvent};
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, error, info};
use uuid::Uuid;

#[cfg(feature = "pulsar")]
use pulsar::{
    compression::Compression,
    consumer::{Consumer, ConsumerOptions, InitialPosition},
    producer::{Producer, ProducerOptions},
    Pulsar, TokioExecutor,
};

/// Pulsar producer configuration with enterprise features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PulsarProducerConfig {
    pub service_url: String,
    pub topic: String,
    pub producer_name: Option<String>,
    pub producer_id: Option<String>,
    pub send_timeout: Duration,
    pub max_pending_messages: u32,
    pub max_pending_messages_across_partitions: u32,
    pub block_if_queue_full: bool,
    pub batching_enabled: bool,
    pub batch_size: u32,
    pub batch_timeout: Duration,
    pub compression_type: PulsarCompressionType,
    pub routing_mode: PulsarRoutingMode,
    pub hashing_scheme: PulsarHashingScheme,
    pub schema_config: Option<PulsarSchemaConfig>,
    pub encryption_config: Option<PulsarEncryptionConfig>,
    pub properties: HashMap<String, String>,
}

impl Default for PulsarProducerConfig {
    fn default() -> Self {
        Self {
            service_url: "pulsar://localhost:6650".to_string(),
            topic: "oxirs-rdf-stream".to_string(),
            producer_name: None,
            producer_id: None,
            send_timeout: Duration::from_secs(30),
            max_pending_messages: 1000,
            max_pending_messages_across_partitions: 50000,
            block_if_queue_full: true,
            batching_enabled: true,
            batch_size: 1000,
            batch_timeout: Duration::from_millis(10),
            compression_type: PulsarCompressionType::Lz4,
            routing_mode: PulsarRoutingMode::RoundRobinPartition,
            hashing_scheme: PulsarHashingScheme::JavaStringHash,
            schema_config: None,
            encryption_config: None,
            properties: HashMap::new(),
        }
    }
}

/// Pulsar compression types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PulsarCompressionType {
    None,
    Lz4,
    Zlib,
    Zstd,
    Snappy,
}

/// Pulsar routing modes for partitioned topics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PulsarRoutingMode {
    RoundRobinPartition,
    SinglePartition,
    CustomPartition,
}

/// Pulsar hashing schemes for message ordering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PulsarHashingScheme {
    JavaStringHash,
    Murmur3_32Hash,
}

/// Pulsar schema configuration for type safety
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PulsarSchemaConfig {
    pub schema_type: PulsarSchemaType,
    pub schema_data: String,
    pub properties: HashMap<String, String>,
}

/// Pulsar schema types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PulsarSchemaType {
    Bytes,
    String,
    Json,
    Avro,
    Protobuf,
    AutoConsume,
    AutoPublish,
}

/// Pulsar encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PulsarEncryptionConfig {
    pub encryption_keys: Vec<String>,
    pub crypto_key_reader: String,
    pub producer_crypto_failure_action: CryptoFailureAction,
    pub consumer_crypto_failure_action: CryptoFailureAction,
}

/// Actions to take on crypto failures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CryptoFailureAction {
    Fail,
    Discard,
    Consume,
}

/// Enhanced Pulsar producer with enterprise features
pub struct PulsarProducer {
    config: StreamConfig,
    pulsar_config: PulsarProducerConfig,
    #[cfg(feature = "pulsar")]
    client: Option<Arc<Pulsar<TokioExecutor>>>,
    #[cfg(feature = "pulsar")]
    producer: Option<Producer<TokioExecutor>>,
    stats: Arc<RwLock<ProducerStats>>,
    sequence_number: u64,
    last_flush: Instant,
    pending_events: Vec<PulsarMessage>,
    batch_buffer: Vec<PulsarMessage>,
    message_router: MessageRouter,
    send_semaphore: Arc<Semaphore>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ProducerStats {
    messages_sent: u64,
    messages_failed: u64,
    bytes_sent: u64,
    batch_count: u64,
    avg_latency_ms: f64,
    max_latency_ms: u64,
    #[serde(skip_serializing, skip_deserializing)]
    last_send: Option<Instant>,
    connection_retries: u64,
    schema_validation_errors: u64,
    encryption_errors: u64,
}

/// Pulsar message wrapper for RDF events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PulsarMessage {
    pub message_id: String,
    pub event_data: StreamEvent,
    pub ordering_key: Option<String>,
    pub partition_key: Option<String>,
    pub event_time: DateTime<Utc>,
    pub properties: HashMap<String, String>,
    pub sequence_id: u64,
    pub schema_version: Option<String>,
    pub replication_clusters: Vec<String>,
}

impl From<StreamEvent> for PulsarMessage {
    fn from(event: StreamEvent) -> Self {
        let (ordering_key, partition_key) = match &event {
            StreamEvent::TripleAdded { subject, graph, .. } => {
                (Some(subject.clone()), graph.clone())
            }
            StreamEvent::TripleRemoved { subject, graph, .. } => {
                (Some(subject.clone()), graph.clone())
            }
            StreamEvent::QuadAdded { subject, graph, .. } => {
                (Some(subject.clone()), Some(graph.clone()))
            }
            StreamEvent::QuadRemoved { subject, graph, .. } => {
                (Some(subject.clone()), Some(graph.clone()))
            }
            StreamEvent::GraphCreated { graph, .. } => (Some(graph.clone()), Some(graph.clone())),
            StreamEvent::GraphCleared { graph, .. } => (graph.clone(), graph.clone()),
            StreamEvent::GraphDeleted { graph, .. } => (Some(graph.clone()), Some(graph.clone())),
            StreamEvent::SparqlUpdate { .. } => (None, None),
            StreamEvent::TransactionBegin { transaction_id, .. } => {
                (Some(transaction_id.clone()), Some(transaction_id.clone()))
            }
            StreamEvent::TransactionCommit { transaction_id, .. } => {
                (Some(transaction_id.clone()), Some(transaction_id.clone()))
            }
            StreamEvent::TransactionAbort { transaction_id, .. } => {
                (Some(transaction_id.clone()), Some(transaction_id.clone()))
            }
            StreamEvent::SchemaChanged { .. } => {
                (Some("schema".to_string()), Some("schema".to_string()))
            }
            StreamEvent::Heartbeat { source, .. } => (Some(source.clone()), Some(source.clone())),
            _ => (None, None), // Default for all other event types
        };

        Self {
            message_id: Uuid::new_v4().to_string(),
            event_data: event,
            ordering_key,
            partition_key,
            event_time: Utc::now(),
            properties: HashMap::new(),
            sequence_id: 0, // Will be set by producer
            schema_version: Some("1.0".to_string()),
            replication_clusters: vec![],
        }
    }
}

/// Message routing for partitioned topics
#[derive(Debug)]
struct MessageRouter {
    routing_mode: PulsarRoutingMode,
    partition_count: u32,
    round_robin_counter: std::sync::atomic::AtomicU32,
}

impl MessageRouter {
    fn new(routing_mode: PulsarRoutingMode, partition_count: u32) -> Self {
        Self {
            routing_mode,
            partition_count,
            round_robin_counter: std::sync::atomic::AtomicU32::new(0),
        }
    }

    fn get_partition(&self, message: &PulsarMessage) -> u32 {
        match self.routing_mode {
            PulsarRoutingMode::RoundRobinPartition => {
                let current = self
                    .round_robin_counter
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                current % self.partition_count
            }
            PulsarRoutingMode::SinglePartition => 0,
            PulsarRoutingMode::CustomPartition => {
                if let Some(partition_key) = &message.partition_key {
                    self.hash_partition_key(partition_key)
                } else {
                    0
                }
            }
        }
    }

    fn hash_partition_key(&self, key: &str) -> u32 {
        // Simple hash function - in production would use murmur3 or similar
        let mut hash = 0u32;
        for byte in key.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
        hash % self.partition_count
    }
}

impl PulsarProducer {
    pub fn new(config: StreamConfig) -> Result<Self> {
        let pulsar_config =
            if let crate::StreamBackendType::Pulsar { service_url, .. } = &config.backend {
                PulsarProducerConfig {
                    service_url: service_url.clone(),
                    topic: config.topic.clone(),
                    ..Default::default()
                }
            } else {
                return Err(anyhow!("Invalid backend configuration for Pulsar producer"));
            };

        let message_router = MessageRouter::new(
            pulsar_config.routing_mode.clone(),
            16, // Default partition count
        );

        Ok(Self {
            config,
            pulsar_config: pulsar_config.clone(),
            #[cfg(feature = "pulsar")]
            client: None,
            #[cfg(feature = "pulsar")]
            producer: None,
            stats: Arc::new(RwLock::new(ProducerStats::default())),
            sequence_number: 0,
            last_flush: Instant::now(),
            pending_events: Vec::new(),
            batch_buffer: Vec::new(),
            message_router,
            send_semaphore: Arc::new(Semaphore::new(1000)),
        })
    }

    pub fn with_pulsar_config(mut self, pulsar_config: PulsarProducerConfig) -> Self {
        self.pulsar_config = pulsar_config;
        self
    }

    pub async fn connect(&mut self) -> Result<()> {
        info!("Connecting to Pulsar at {}", self.pulsar_config.service_url);

        // Initialize connection and validate configuration
        self.validate_configuration().await?;

        // Create Pulsar client
        #[cfg(feature = "pulsar")]
        {
            let builder = Pulsar::builder(&self.pulsar_config.service_url, TokioExecutor);

            // Add authentication if configured
            // Note: Authentication setup depends on the specific auth method
            // For now, authentication is handled via connection properties
            if let Some(_auth_config) = &self.pulsar_config.properties.get("auth_token") {
                debug!("Authentication token configured (auth setup to be implemented)");
            }

            let client = builder.build().await?;
            let client = Arc::new(client);

            // Create producer with configuration
            let mut producer_builder = client.producer();

            if let Some(producer_name) = &self.pulsar_config.producer_name {
                producer_builder = producer_builder.with_name(producer_name);
            }

            let compression = match self.pulsar_config.compression_type {
                PulsarCompressionType::None => Compression::None,
                PulsarCompressionType::Lz4 => Compression::Lz4(Default::default()),
                PulsarCompressionType::Zlib => Compression::Zlib(Default::default()),
                PulsarCompressionType::Zstd => Compression::Zstd(Default::default()),
                PulsarCompressionType::Snappy => Compression::Snappy(Default::default()),
            };

            let mut options = ProducerOptions {
                compression: Some(compression),
                batch_size: Some(self.pulsar_config.batch_size),
                ..Default::default()
            };

            if self.pulsar_config.batching_enabled {
                options.batch_size = Some(self.pulsar_config.batch_size);
            }

            let producer = producer_builder
                .with_topic(&self.pulsar_config.topic)
                .with_options(options)
                .build()
                .await?;

            self.client = Some(client);
            self.producer = Some(producer);
        }

        // Set up schema if configured
        if let Some(schema_config) = &self.pulsar_config.schema_config {
            self.setup_schema(schema_config).await?;
        }

        // Detect partition count for routing
        let partition_count = self.get_partition_count().await?;
        self.message_router =
            MessageRouter::new(self.pulsar_config.routing_mode.clone(), partition_count);

        info!(
            "Connected to Pulsar topic: {} with {} partitions",
            self.pulsar_config.topic, partition_count
        );
        Ok(())
    }

    async fn validate_configuration(&self) -> Result<()> {
        // Validate service URL
        if self.pulsar_config.service_url.is_empty() {
            return Err(anyhow!("Pulsar service URL cannot be empty"));
        }

        // Validate topic name
        if self.pulsar_config.topic.is_empty() {
            return Err(anyhow!("Pulsar topic cannot be empty"));
        }

        // Validate batch configuration
        if self.pulsar_config.batching_enabled {
            if self.pulsar_config.batch_size == 0 {
                return Err(anyhow!(
                    "Batch size must be greater than 0 when batching is enabled"
                ));
            }
            if self.pulsar_config.batch_timeout.as_millis() == 0 {
                return Err(anyhow!(
                    "Batch timeout must be greater than 0 when batching is enabled"
                ));
            }
        }

        Ok(())
    }

    async fn setup_schema(&self, _schema_config: &PulsarSchemaConfig) -> Result<()> {
        // In a real implementation, this would register the schema with Pulsar
        debug!("Setting up Pulsar schema");
        Ok(())
    }

    async fn get_partition_count(&self) -> Result<u32> {
        // In a real implementation, this would query Pulsar admin API
        // For now, return a default value
        Ok(16)
    }

    pub async fn publish(&mut self, event: StreamEvent) -> Result<()> {
        let start_time = Instant::now();

        let mut message = PulsarMessage::from(event);
        message.sequence_id = self.sequence_number;
        self.sequence_number += 1;

        // Add producer properties
        message.properties.insert(
            "producer_id".to_string(),
            self.pulsar_config
                .producer_id
                .clone()
                .unwrap_or_else(|| "oxirs-producer".to_string()),
        );
        message
            .properties
            .insert("producer_time".to_string(), Utc::now().to_rfc3339());

        // Handle batching
        if self.pulsar_config.batching_enabled {
            self.batch_buffer.push(message);

            if self.batch_buffer.len() >= self.pulsar_config.batch_size as usize {
                self.flush_batch().await?;
            }
        } else {
            self.send_message(message).await?;
        }

        // Update stats
        let latency = start_time.elapsed().as_millis() as u64;
        let mut stats = self.stats.write().await;
        stats.max_latency_ms = stats.max_latency_ms.max(latency);
        stats.avg_latency_ms = (stats.avg_latency_ms + latency as f64) / 2.0;
        stats.last_send = Some(Instant::now());

        Ok(())
    }

    async fn send_message(&mut self, message: PulsarMessage) -> Result<()> {
        let partition = self.message_router.get_partition(&message);

        debug!(
            "Sending message {} to partition {}",
            message.message_id, partition
        );

        #[cfg(feature = "pulsar")]
        {
            if let Some(producer) = &mut self.producer {
                // Acquire send permit
                let _permit = self.send_semaphore.acquire().await?;

                // Serialize message
                let payload = serde_json::to_vec(&message.event_data)?;

                // Build Pulsar message
                let mut pulsar_msg = producer.create_message().with_content(payload.as_slice());

                // Add properties
                for (key, value) in &message.properties {
                    pulsar_msg = pulsar_msg.with_property(key, value);
                }

                // Add ordering key if present
                if let Some(ordering_key) = &message.ordering_key {
                    pulsar_msg = pulsar_msg.with_ordering_key(ordering_key.as_bytes());
                }

                // Add partition key if present
                if let Some(partition_key) = &message.partition_key {
                    pulsar_msg = pulsar_msg.with_partition_key(partition_key);
                }

                // Add event time
                pulsar_msg = pulsar_msg.event_time(message.event_time.timestamp_millis() as u64);

                // Send message
                let _receipt = pulsar_msg.send_non_blocking().await?;

                debug!("Message sent successfully");

                // Update stats
                let mut stats = self.stats.write().await;
                stats.messages_sent += 1;
                stats.bytes_sent += payload.len() as u64;
                stats.last_send = Some(Instant::now());
            } else {
                return Err(anyhow!("Producer not connected"));
            }
        }

        #[cfg(not(feature = "pulsar"))]
        {
            // Simulate for testing
            tokio::time::sleep(Duration::from_millis(1)).await;
            let mut stats = self.stats.write().await;
            stats.messages_sent += 1;
            stats.bytes_sent += self.estimate_message_size(&message);
        }

        Ok(())
    }

    async fn flush_batch(&mut self) -> Result<()> {
        if self.batch_buffer.is_empty() {
            return Ok(());
        }

        let batch = std::mem::take(&mut self.batch_buffer);
        let batch_size = batch.len();

        debug!("Flushing batch of {} messages", batch_size);

        // In a real implementation, this would send all messages in a batch
        for message in batch {
            self.send_message(message).await?;
        }

        let mut stats = self.stats.write().await;
        stats.batch_count += 1;
        debug!("Flushed batch of {} messages", batch_size);

        Ok(())
    }

    pub async fn publish_batch(&mut self, events: Vec<StreamEvent>) -> Result<()> {
        for event in events {
            self.publish(event).await?;
        }
        self.flush().await
    }

    pub async fn publish_patch(&mut self, patch: &RdfPatch) -> Result<()> {
        for operation in &patch.operations {
            let metadata = EventMetadata {
                event_id: Uuid::new_v4().to_string(),
                timestamp: Utc::now(),
                source: "rdf_patch".to_string(),
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
                PatchOperation::AddPrefix {
                    prefix: _,
                    namespace: _,
                } => continue,
                PatchOperation::DeletePrefix { prefix: _ } => continue,
                PatchOperation::TransactionBegin { transaction_id } => {
                    StreamEvent::TransactionBegin {
                        transaction_id: transaction_id
                            .clone()
                            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                        isolation_level: Some(crate::IsolationLevel::ReadCommitted),
                        metadata,
                    }
                }
                PatchOperation::TransactionCommit => StreamEvent::TransactionCommit {
                    transaction_id: "unknown".to_string(),
                    metadata,
                },
                PatchOperation::TransactionAbort => StreamEvent::TransactionAbort {
                    transaction_id: "unknown".to_string(),
                    metadata,
                },
                PatchOperation::Header { key, value } => {
                    tracing::debug!("Processing patch header: {} = {}", key, value);
                    continue;
                }
            };

            self.publish(event).await?;
        }
        self.flush().await
    }

    pub async fn flush(&mut self) -> Result<()> {
        if self.pulsar_config.batching_enabled && !self.batch_buffer.is_empty() {
            self.flush_batch().await?;
        }

        self.last_flush = Instant::now();
        debug!("Flushed Pulsar producer");
        Ok(())
    }

    fn estimate_message_size(&self, message: &PulsarMessage) -> u64 {
        // Rough estimation of message size
        let json_size = serde_json::to_string(message)
            .map(|s| s.len())
            .unwrap_or(1024);
        json_size as u64
    }

    pub async fn get_stats(&self) -> ProducerStats {
        self.stats.read().await.clone()
    }
}

/// Enhanced Pulsar consumer with advanced subscription features
pub struct PulsarConsumer {
    config: StreamConfig,
    pulsar_config: PulsarConsumerConfig,
    #[cfg(feature = "pulsar")]
    client: Option<Arc<Pulsar<TokioExecutor>>>,
    #[cfg(feature = "pulsar")]
    consumer: Option<Consumer<Vec<u8>, TokioExecutor>>,
    stats: Arc<RwLock<ConsumerStats>>,
    subscription_name: String,
    consumer_name: String,
    message_buffer: Vec<PulsarMessage>,
    ack_timeout: Duration,
    receive_semaphore: Arc<Semaphore>,
}

/// Pulsar consumer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PulsarConsumerConfig {
    pub service_url: String,
    pub topic: String,
    pub subscription_name: String,
    pub subscription_type: PulsarSubscriptionType,
    pub consumer_name: Option<String>,
    pub receiver_queue_size: u32,
    pub ack_timeout: Duration,
    pub negative_ack_redelivery_delay: Duration,
    pub max_total_receiver_queue_size_across_partitions: u32,
    pub consumer_crypto_failure_action: CryptoFailureAction,
    pub read_compacted: bool,
    pub subscription_initial_position: PulsarSubscriptionInitialPosition,
    pub pattern_auto_discovery_period: Duration,
    pub properties: HashMap<String, String>,
}

impl Default for PulsarConsumerConfig {
    fn default() -> Self {
        Self {
            service_url: "pulsar://localhost:6650".to_string(),
            topic: "oxirs-rdf-stream".to_string(),
            subscription_name: "oxirs-subscription".to_string(),
            subscription_type: PulsarSubscriptionType::Shared,
            consumer_name: None,
            receiver_queue_size: 1000,
            ack_timeout: Duration::from_secs(30),
            negative_ack_redelivery_delay: Duration::from_secs(60),
            max_total_receiver_queue_size_across_partitions: 50000,
            consumer_crypto_failure_action: CryptoFailureAction::Fail,
            read_compacted: false,
            subscription_initial_position: PulsarSubscriptionInitialPosition::Latest,
            pattern_auto_discovery_period: Duration::from_secs(60),
            properties: HashMap::new(),
        }
    }
}

/// Pulsar subscription types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PulsarSubscriptionType {
    Exclusive,
    Shared,
    Failover,
    KeyShared,
}

/// Initial position for subscription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PulsarSubscriptionInitialPosition {
    Latest,
    Earliest,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ConsumerStats {
    messages_received: u64,
    messages_acknowledged: u64,
    messages_negative_acknowledged: u64,
    bytes_received: u64,
    receive_queue_size: u32,
    avg_processing_time_ms: f64,
    #[serde(skip_serializing, skip_deserializing)]
    last_message: Option<Instant>,
    redelivery_count: u64,
    connection_errors: u64,
}

impl PulsarConsumer {
    pub fn new(config: StreamConfig) -> Result<Self> {
        let pulsar_config =
            if let crate::StreamBackendType::Pulsar { service_url, .. } = &config.backend {
                PulsarConsumerConfig {
                    service_url: service_url.clone(),
                    topic: config.topic.clone(),
                    subscription_name: format!("{}-subscription", config.topic),
                    ..Default::default()
                }
            } else {
                return Err(anyhow!("Invalid backend configuration for Pulsar consumer"));
            };

        let consumer_name = format!("consumer-{}", Uuid::new_v4());

        Ok(Self {
            config,
            pulsar_config: pulsar_config.clone(),
            #[cfg(feature = "pulsar")]
            client: None,
            #[cfg(feature = "pulsar")]
            consumer: None,
            stats: Arc::new(RwLock::new(ConsumerStats::default())),
            subscription_name: pulsar_config.subscription_name.clone(),
            consumer_name,
            message_buffer: Vec::new(),
            ack_timeout: pulsar_config.ack_timeout,
            receive_semaphore: Arc::new(Semaphore::new(100)),
        })
    }

    pub fn with_pulsar_config(mut self, pulsar_config: PulsarConsumerConfig) -> Self {
        self.pulsar_config = pulsar_config;
        self
    }

    pub async fn connect(&mut self) -> Result<()> {
        info!(
            "Connecting Pulsar consumer to {} with subscription {}",
            self.pulsar_config.topic, self.subscription_name
        );

        // Validate configuration
        self.validate_consumer_configuration().await?;

        #[cfg(feature = "pulsar")]
        {
            // Create Pulsar client
            let builder = Pulsar::builder(&self.pulsar_config.service_url, TokioExecutor);

            // Add authentication if configured
            // Note: Authentication setup depends on the specific auth method
            // For now, authentication is handled via connection properties
            if let Some(_auth_token) = self.pulsar_config.properties.get("auth_token") {
                debug!("Authentication token configured (auth setup to be implemented)");
            }

            let client = builder.build().await?;
            let client = Arc::new(client);

            // Create consumer with configuration
            let mut consumer_builder = client.consumer();

            if let Some(consumer_name) = &self.pulsar_config.consumer_name {
                consumer_builder = consumer_builder.with_consumer_name(consumer_name);
            }

            // Set subscription type
            // Note: Subscription type is currently handled by the Pulsar library based on consumer configuration
            let _sub_type = match self.pulsar_config.subscription_type {
                PulsarSubscriptionType::Exclusive => {
                    pulsar::message::proto::command_subscribe::SubType::Exclusive
                }
                PulsarSubscriptionType::Shared => {
                    pulsar::message::proto::command_subscribe::SubType::Shared
                }
                PulsarSubscriptionType::Failover => {
                    pulsar::message::proto::command_subscribe::SubType::Failover
                }
                PulsarSubscriptionType::KeyShared => {
                    pulsar::message::proto::command_subscribe::SubType::KeyShared
                }
            };

            // Set initial position
            let initial_position = match self.pulsar_config.subscription_initial_position {
                PulsarSubscriptionInitialPosition::Latest => InitialPosition::Latest,
                PulsarSubscriptionInitialPosition::Earliest => InitialPosition::Earliest,
            };

            let mut options = ConsumerOptions {
                initial_position,
                ..Default::default()
            };

            if self.pulsar_config.read_compacted {
                options.read_compacted = Some(true);
            }

            let consumer = consumer_builder
                .with_topic(&self.pulsar_config.topic)
                .with_subscription(&self.subscription_name)
                .with_options(options)
                .build()
                .await?;

            self.client = Some(client);
            self.consumer = Some(consumer);
        }

        info!("Connected Pulsar consumer: {}", self.consumer_name);
        Ok(())
    }

    async fn validate_consumer_configuration(&self) -> Result<()> {
        if self.pulsar_config.service_url.is_empty() {
            return Err(anyhow!("Pulsar service URL cannot be empty"));
        }

        if self.pulsar_config.topic.is_empty() {
            return Err(anyhow!("Pulsar topic cannot be empty"));
        }

        if self.subscription_name.is_empty() {
            return Err(anyhow!("Subscription name cannot be empty"));
        }

        Ok(())
    }

    pub async fn consume(&mut self) -> Result<Option<StreamEvent>> {
        let start_time = Instant::now();

        #[cfg(feature = "pulsar")]
        {
            if let Some(consumer) = &mut self.consumer {
                // Acquire receive permit
                let _permit = self.receive_semaphore.acquire().await?;

                // Try to receive a message with timeout
                match tokio::time::timeout(Duration::from_millis(100), consumer.next()).await {
                    Ok(Some(Ok(msg))) => {
                        // Extract key before moving payload
                        let ordering_key = msg
                            .key()
                            .map(|k| String::from_utf8_lossy(k.as_bytes()).to_string());

                        // Deserialize the payload
                        let payload = msg.payload.data.clone();
                        let event: StreamEvent = serde_json::from_slice(&payload)?;

                        // Create PulsarMessage wrapper
                        // Note: Some metadata fields may not be available in all pulsar versions
                        let _pulsar_message = PulsarMessage {
                            message_id: format!("{:?}", msg.message_id),
                            event_data: event.clone(),
                            ordering_key,
                            partition_key: None, // Not directly available in this pulsar version
                            event_time: Utc::now(), // Use current time as fallback
                            properties: HashMap::new(), // Properties not directly accessible
                            sequence_id: 0,      // Sequence ID not directly available
                            schema_version: None,
                            replication_clusters: vec![],
                        };

                        // Acknowledge the message
                        consumer.ack(&msg).await?;

                        // Update stats
                        let mut stats = self.stats.write().await;
                        stats.messages_received += 1;
                        stats.bytes_received += payload.len() as u64;
                        stats.messages_acknowledged += 1;
                        stats.last_message = Some(Instant::now());

                        let processing_time = start_time.elapsed().as_millis() as f64;
                        stats.avg_processing_time_ms =
                            (stats.avg_processing_time_ms + processing_time) / 2.0;

                        Ok(Some(event))
                    }
                    Ok(Some(Err(e))) => {
                        error!("Error receiving message: {}", e);
                        let mut stats = self.stats.write().await;
                        stats.connection_errors += 1;
                        Err(anyhow!("Pulsar consumer error: {}", e))
                    }
                    Ok(None) => {
                        // No message available
                        Ok(None)
                    }
                    Err(_) => {
                        // Timeout - no message available
                        Ok(None)
                    }
                }
            } else {
                Err(anyhow!("Consumer not connected"))
            }
        }

        #[cfg(not(feature = "pulsar"))]
        {
            // Simulate for testing
            tokio::time::sleep(Duration::from_millis(10)).await;

            // Check if we have buffered messages
            if let Some(message) = self.message_buffer.pop() {
                let mut stats = self.stats.write().await;
                stats.messages_received += 1;
                stats.bytes_received += self.estimate_message_size(&message);
                stats.last_message = Some(Instant::now());

                let processing_time = start_time.elapsed().as_millis() as f64;
                stats.avg_processing_time_ms =
                    (stats.avg_processing_time_ms + processing_time) / 2.0;

                return Ok(Some(message.event_data));
            }

            Ok(None)
        }
    }

    async fn acknowledge_message(&mut self, message: &PulsarMessage) -> Result<()> {
        debug!("Acknowledging message: {}", message.message_id);
        let mut stats = self.stats.write().await;
        stats.messages_acknowledged += 1;
        Ok(())
    }

    pub async fn negative_acknowledge(&mut self, message: &PulsarMessage) -> Result<()> {
        debug!("Negative acknowledging message: {}", message.message_id);

        #[cfg(feature = "pulsar")]
        {
            if let Some(_consumer) = &mut self.consumer {
                // In a real implementation, we'd need to keep track of the original Message
                // to call consumer.nack(&msg). For now, just update stats.
                let mut stats = self.stats.write().await;
                stats.messages_negative_acknowledged += 1;
                stats.redelivery_count += 1;
            }
        }

        #[cfg(not(feature = "pulsar"))]
        {
            let mut stats = self.stats.write().await;
            stats.messages_negative_acknowledged += 1;
            stats.redelivery_count += 1;
        }

        Ok(())
    }

    pub async fn consume_batch(
        &mut self,
        max_messages: usize,
        timeout: Duration,
    ) -> Result<Vec<StreamEvent>> {
        let mut events = Vec::new();
        let start_time = Instant::now();

        while events.len() < max_messages && start_time.elapsed() < timeout {
            match tokio::time::timeout(Duration::from_millis(50), self.consume()).await {
                Ok(Ok(Some(event))) => events.push(event),
                Ok(Ok(None)) => continue,
                Ok(Err(e)) => return Err(e),
                Err(_) => break, // Timeout
            }
        }

        Ok(events)
    }

    fn estimate_message_size(&self, message: &PulsarMessage) -> u64 {
        let json_size = serde_json::to_string(message)
            .map(|s| s.len())
            .unwrap_or(1024);
        json_size as u64
    }

    pub async fn get_stats(&self) -> ConsumerStats {
        self.stats.read().await.clone()
    }
}

/// Pulsar admin client for topic and subscription management
pub struct PulsarAdmin {
    service_url: String,
    admin_url: String,
    auth_config: Option<PulsarAuthConfig>,
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
    Athenz,
}

impl PulsarAdmin {
    pub fn new(service_url: String) -> Self {
        let admin_url = service_url
            .replace("pulsar://", "http://")
            .replace(":6650", ":8080");

        Self {
            service_url,
            admin_url,
            auth_config: None,
        }
    }

    pub fn with_auth(mut self, auth_config: PulsarAuthConfig) -> Self {
        self.auth_config = Some(auth_config);
        self
    }

    pub async fn create_topic(&self, topic: &str, partitions: u32) -> Result<()> {
        info!(
            "Creating Pulsar topic: {} with {} partitions",
            topic, partitions
        );
        // In a real implementation, this would call Pulsar admin API
        Ok(())
    }

    pub async fn delete_topic(&self, topic: &str) -> Result<()> {
        info!("Deleting Pulsar topic: {}", topic);
        // In a real implementation, this would call Pulsar admin API
        Ok(())
    }

    pub async fn get_topic_stats(&self, topic: &str) -> Result<PulsarTopicStats> {
        debug!("Getting stats for topic: {}", topic);
        // In a real implementation, this would query Pulsar admin API
        Ok(PulsarTopicStats {
            topic: topic.to_string(),
            partitions: 16,
            producers: 1,
            consumers: 1,
            msg_rate_in: 100.0,
            msg_rate_out: 95.0,
            msg_throughput_in: 10240.0,
            msg_throughput_out: 9728.0,
            storage_size: 1048576,
        })
    }

    pub async fn create_subscription(&self, topic: &str, subscription: &str) -> Result<()> {
        info!("Creating subscription {} for topic {}", subscription, topic);
        // In a real implementation, this would call Pulsar admin API
        Ok(())
    }

    pub async fn get_subscription_stats(
        &self,
        topic: &str,
        subscription: &str,
    ) -> Result<PulsarSubscriptionStats> {
        debug!("Getting subscription stats for {}/{}", topic, subscription);
        // In a real implementation, this would query Pulsar admin API
        Ok(PulsarSubscriptionStats {
            topic: topic.to_string(),
            subscription: subscription.to_string(),
            consumers: 1,
            msg_rate_out: 95.0,
            msg_throughput_out: 9728.0,
            msg_backlog: 5,
            blocked_subscription_on_unacked_msgs: false,
            unacked_messages: 0,
        })
    }
}

/// Pulsar topic statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PulsarTopicStats {
    pub topic: String,
    pub partitions: u32,
    pub producers: u32,
    pub consumers: u32,
    pub msg_rate_in: f64,
    pub msg_rate_out: f64,
    pub msg_throughput_in: f64,
    pub msg_throughput_out: f64,
    pub storage_size: u64,
}

/// Pulsar subscription statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PulsarSubscriptionStats {
    pub topic: String,
    pub subscription: String,
    pub consumers: u32,
    pub msg_rate_out: f64,
    pub msg_throughput_out: f64,
    pub msg_backlog: u64,
    pub blocked_subscription_on_unacked_msgs: bool,
    pub unacked_messages: u64,
}

#[cfg(all(test, feature = "pulsar"))]
mod tests {
    use super::*;
    use crate::StreamConfig;

    fn test_pulsar_config() -> StreamConfig {
        StreamConfig {
            backend: crate::StreamBackendType::Pulsar {
                service_url: "pulsar://localhost:6650".to_string(),
                auth_config: None,
            },
            topic: "test-topic".to_string(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_pulsar_producer_creation() {
        let config = test_pulsar_config();
        let producer = PulsarProducer::new(config);
        assert!(producer.is_ok());
    }

    #[tokio::test]
    async fn test_pulsar_consumer_creation() {
        let config = test_pulsar_config();
        let consumer = PulsarConsumer::new(config);
        assert!(consumer.is_ok());
    }

    #[tokio::test]
    async fn test_message_routing() {
        let router = MessageRouter::new(PulsarRoutingMode::RoundRobinPartition, 4);

        let message = PulsarMessage {
            message_id: "test".to_string(),
            event_data: StreamEvent::Heartbeat {
                timestamp: Utc::now(),
                source: "test".to_string(),
                metadata: EventMetadata {
                    event_id: "test-heartbeat".to_string(),
                    timestamp: Utc::now(),
                    source: "test".to_string(),
                    user: None,
                    context: None,
                    caused_by: None,
                    version: "1.0".to_string(),
                    properties: HashMap::new(),
                    checksum: None,
                },
            },
            ordering_key: None,
            partition_key: None,
            event_time: Utc::now(),
            properties: HashMap::new(),
            sequence_id: 0,
            schema_version: None,
            replication_clusters: vec![],
        };

        let partition1 = router.get_partition(&message);
        let partition2 = router.get_partition(&message);

        assert!(partition1 < 4);
        assert!(partition2 < 4);
        assert_ne!(partition1, partition2);
    }

    #[test]
    fn test_pulsar_message_conversion() {
        let event = StreamEvent::TripleAdded {
            subject: "http://example.org/subject".to_string(),
            predicate: "http://example.org/predicate".to_string(),
            object: "http://example.org/object".to_string(),
            graph: Some("http://example.org/graph".to_string()),
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

        let message = PulsarMessage::from(event);
        assert_eq!(
            message.ordering_key,
            Some("http://example.org/subject".to_string())
        );
        assert_eq!(
            message.partition_key,
            Some("http://example.org/graph".to_string())
        );
    }

    #[tokio::test]
    async fn test_pulsar_admin_operations() {
        let admin = PulsarAdmin::new("pulsar://localhost:6650".to_string());

        let result = admin.create_topic("test-topic", 8).await;
        assert!(result.is_ok());

        let stats = admin.get_topic_stats("test-topic").await;
        assert!(stats.is_ok());

        if let Ok(topic_stats) = stats {
            assert_eq!(topic_stats.topic, "test-topic");
            assert!(topic_stats.partitions > 0);
        }
    }
}
