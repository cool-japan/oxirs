//! # Apache Kafka Backend - Modular Architecture
//!
//! Complete Apache Kafka integration for enterprise-scale RDF streaming.
//!
//! This module provides comprehensive Kafka integration with transactional producers,
//! exactly-once semantics, schema registry, consumer groups, and advanced performance
//! optimizations for mission-critical RDF streaming applications.
//!
//! ## Architecture
//!
//! The Kafka backend is organized into specialized modules:
//! - `config`: Configuration management and security settings
//! - `message`: Message types and serialization
//! - `producer`: Transactional producer implementation
//! - `consumer`: Consumer group management
//! - `admin`: Topic and cluster administration

// Import modular components from local modules
use super::{KafkaEvent, KafkaProducerConfig, KafkaProducerStats};

use crate::backend::StreamBackend as StreamBackendTrait;
use crate::consumer::ConsumerGroup;
use crate::error::{StreamError, StreamResult};
use crate::types::{Offset, PartitionId, StreamPosition, TopicName};
use crate::{StreamConfig, StreamEvent};
use anyhow::Result;
use async_trait::async_trait;
use rdkafka::producer::Producer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokio::sync::{oneshot, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

#[cfg(feature = "kafka")]
use rdkafka::{
    admin::{AdminClient, AdminOptions, NewTopic, TopicReplication},
    config::ClientConfig,
    consumer::{BaseConsumer, Consumer, ConsumerContext, Rebalance, StreamConsumer},
    message::{BorrowedMessage, Message},
    producer::{FutureProducer, FutureRecord},
    ClientContext,
};

/// Consumer identifier type
pub type ConsumerId = Uuid;

/// Message callback type for streaming consumers
pub type MessageCallback = Arc<dyn Fn(StreamEvent) -> Result<()> + Send + Sync>;

/// Consumer instance management
struct ConsumerInstance {
    id: ConsumerId,
    group_id: String,
    #[cfg(feature = "kafka")]
    consumer: Arc<StreamConsumer>,
    is_active: Arc<AtomicBool>,
    message_count: Arc<AtomicU64>,
    error_count: Arc<AtomicU64>,
    last_message_time: Arc<RwLock<Option<chrono::DateTime<chrono::Utc>>>>,
    stop_signal: Option<oneshot::Sender<()>>,
}

/// Consumer metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumerMetrics {
    pub consumer_id: ConsumerId,
    pub group_id: String,
    pub messages_processed: u64,
    pub errors_encountered: u64,
    pub is_active: bool,
    pub last_message_time: Option<chrono::DateTime<chrono::Utc>>,
    pub partition_assignments: Vec<PartitionAssignment>,
}

/// Partition assignment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionAssignment {
    pub topic: String,
    pub partition: i32,
    pub current_offset: i64,
    pub high_water_mark: i64,
    pub lag: i64,
}

/// Consumer rebalance callback context
#[cfg(feature = "kafka")]
struct ConsumerRebalanceContext {
    consumer_id: ConsumerId,
}

#[cfg(feature = "kafka")]
impl ClientContext for ConsumerRebalanceContext {}

#[cfg(feature = "kafka")]
impl ConsumerContext for ConsumerRebalanceContext {
    fn pre_rebalance(&self, _consumer: &BaseConsumer<Self>, rebalance: &Rebalance) {
        info!(
            "Consumer {} pre-rebalance: {:?}",
            self.consumer_id, rebalance
        );
    }

    fn post_rebalance(&self, _consumer: &BaseConsumer<Self>, rebalance: &Rebalance) {
        info!(
            "Consumer {} post-rebalance: {:?}",
            self.consumer_id, rebalance
        );
    }

    fn commit_callback(
        &self,
        result: rdkafka::error::KafkaResult<()>,
        offsets: &rdkafka::TopicPartitionList,
    ) {
        match result {
            Ok(_) => debug!(
                "Consumer {} committed offsets: {:?}",
                self.consumer_id, offsets
            ),
            Err(e) => error!("Consumer {} commit failed: {}", self.consumer_id, e),
        }
    }
}

/// Kafka backend implementation that coordinates all modules
pub struct KafkaBackend {
    config: StreamConfig,
    kafka_config: KafkaProducerConfig,
    #[cfg(feature = "kafka")]
    producer: Option<FutureProducer>,
    #[cfg(feature = "kafka")]
    consumer: Option<StreamConsumer>,
    #[cfg(feature = "kafka")]
    admin_client: Option<AdminClient<rdkafka::client::DefaultClientContext>>,
    #[cfg(not(feature = "kafka"))]
    _phantom: std::marker::PhantomData<()>,
    stats: Arc<RwLock<KafkaProducerStats>>,
    // Consumer management
    active_consumers: Arc<RwLock<HashMap<ConsumerId, ConsumerInstance>>>,
}

impl KafkaBackend {
    /// Create a new Kafka backend with the given configuration
    pub fn new(config: StreamConfig) -> Result<Self> {
        let kafka_config = if let crate::StreamBackendType::Kafka { brokers, .. } = &config.backend
        {
            KafkaProducerConfig {
                brokers: brokers.clone(),
                ..Default::default()
            }
        } else {
            KafkaProducerConfig::default()
        };

        Ok(Self {
            config,
            kafka_config,
            #[cfg(feature = "kafka")]
            producer: None,
            #[cfg(feature = "kafka")]
            consumer: None,
            #[cfg(feature = "kafka")]
            admin_client: None,
            #[cfg(not(feature = "kafka"))]
            _phantom: std::marker::PhantomData,
            stats: Arc::new(RwLock::new(KafkaProducerStats::default())),
            active_consumers: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Configure the backend with custom Kafka settings
    pub fn with_kafka_config(mut self, kafka_config: KafkaProducerConfig) -> Self {
        self.kafka_config = kafka_config;
        self
    }

    /// Initialize the Kafka backend connections
    #[cfg(feature = "kafka")]
    pub async fn connect(&mut self) -> Result<()> {
        let mut client_config = ClientConfig::new();

        // Apply configuration from KafkaProducerConfig
        self.apply_client_config(&mut client_config);

        // Create producer
        let producer: FutureProducer = client_config
            .create()
            .map_err(|e| anyhow::anyhow!("Failed to create Kafka producer: {}", e))?;

        // Create consumer for receiving events if needed
        let consumer: StreamConsumer = client_config
            .create()
            .map_err(|e| anyhow::anyhow!("Failed to create Kafka consumer: {}", e))?;

        // Create admin client for topic management
        let admin_client: AdminClient<rdkafka::client::DefaultClientContext> = client_config
            .create()
            .map_err(|e| anyhow::anyhow!("Failed to create Kafka admin client: {}", e))?;

        self.producer = Some(producer);
        self.consumer = Some(consumer);
        self.admin_client = Some(admin_client);

        info!(
            "Connected to Kafka cluster at {}",
            self.kafka_config.brokers.join(",")
        );
        Ok(())
    }

    /// Apply configuration to the Kafka client
    #[cfg(feature = "kafka")]
    fn apply_client_config(&self, client_config: &mut ClientConfig) {
        client_config
            .set("bootstrap.servers", self.kafka_config.brokers.join(","))
            .set("client.id", &self.kafka_config.client_id)
            .set("acks", self.kafka_config.acks.to_string())
            .set("retries", self.kafka_config.retries.to_string())
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
            );

        // Enable idempotence for exactly-once semantics
        if self.kafka_config.enable_idempotence {
            client_config.set("enable.idempotence", "true");
        }

        // Transaction ID for transactional producer
        if let Some(ref transaction_id) = self.kafka_config.transaction_id {
            client_config.set("transactional.id", transaction_id);
        }

        // Apply security configuration
        if let Some(ref security_config) = self.kafka_config.security_config {
            client_config.set(
                "security.protocol",
                security_config.security_protocol.to_string(),
            );

            if let Some(ref sasl_config) = security_config.sasl_config {
                client_config
                    .set("sasl.mechanism", sasl_config.mechanism.to_string())
                    .set("sasl.username", &sasl_config.username)
                    .set("sasl.password", &sasl_config.password);
            }
        }
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn connect(&mut self) -> Result<()> {
        warn!("Kafka feature not enabled, using mock backend");
        Ok(())
    }

    /// Create a topic with specified configuration
    #[cfg(feature = "kafka")]
    pub async fn create_topic(
        &self,
        topic_name: &str,
        partitions: i32,
        replication_factor: i16,
    ) -> Result<()> {
        if let Some(ref admin_client) = self.admin_client {
            let new_topic = NewTopic::new(
                topic_name,
                partitions,
                TopicReplication::Fixed(replication_factor.into()),
            );
            let opts = AdminOptions::new().operation_timeout(Some(Duration::from_secs(30)));

            match admin_client.create_topics(&[new_topic], &opts).await {
                Ok(results) => {
                    for result in results {
                        match result {
                            Ok(topic) => info!("Created topic: {}", topic),
                            Err((topic, error)) => {
                                if error.to_string().contains("already exists") {
                                    debug!("Topic {} already exists", topic);
                                } else {
                                    error!("Failed to create topic {}: {}", topic, error);
                                    return Err(anyhow::anyhow!(
                                        "Failed to create topic: {}",
                                        error
                                    ));
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    return Err(anyhow::anyhow!("Failed to create topics: {}", e));
                }
            }
        }
        Ok(())
    }

    /// Publish a single event to Kafka
    pub async fn publish(&mut self, event: StreamEvent) -> Result<()> {
        let kafka_event = KafkaEvent::from(event);
        let topic_name = kafka_event.get_topic_name("rdf");
        let start_time = Instant::now();

        #[cfg(feature = "kafka")]
        {
            if self.producer.is_none() {
                self.connect().await?;
            }

            if let Some(ref producer) = self.producer {
                let payload = kafka_event
                    .to_bytes()
                    .map_err(|e| anyhow::anyhow!("Failed to serialize event: {}", e))?;

                let mut record = FutureRecord::to(&topic_name)
                    .payload(&payload)
                    .key(&kafka_event.event_id);

                // Set partition key if available
                if let Some(ref partition_key) = kafka_event.partition_key {
                    record = record.key(partition_key);
                }

                match producer.send(record, Duration::from_secs(30)).await {
                    Ok(_delivery) => {
                        let latency_ms = start_time.elapsed().as_millis() as u64;
                        self.update_stats(payload.len(), latency_ms, false).await;
                        debug!(
                            "Published event to Kafka: {} to topic {}",
                            kafka_event.event_id, topic_name
                        );
                    }
                    Err((kafka_error, _)) => {
                        self.update_stats(0, 0, true).await;
                        error!("Failed to publish to Kafka: {}", kafka_error);
                        return Err(anyhow::anyhow!("Kafka publish failed: {}", kafka_error));
                    }
                }
            } else {
                return Err(anyhow::anyhow!("Kafka producer not initialized"));
            }
        }
        #[cfg(not(feature = "kafka"))]
        {
            debug!(
                "Mock Kafka publish: {} to topic {}",
                kafka_event.event_id, topic_name
            );
        }

        Ok(())
    }

    /// Update internal statistics
    async fn update_stats(&self, bytes_sent: usize, latency_ms: u64, is_error: bool) {
        let mut stats = self.stats.write().await;
        if is_error {
            stats.events_failed += 1;
            stats.delivery_errors += 1;
        } else {
            stats.events_published += 1;
            stats.bytes_sent += bytes_sent as u64;
            stats.last_publish = Some(chrono::Utc::now());

            // Update latency metrics
            if latency_ms > stats.max_latency_ms {
                stats.max_latency_ms = latency_ms;
            }

            // Simple running average
            let total_events = stats.events_published as f64;
            stats.avg_latency_ms =
                (stats.avg_latency_ms * (total_events - 1.0) + latency_ms as f64) / total_events;
        }
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> KafkaProducerStats {
        self.stats.read().await.clone()
    }

    /// Publish multiple events in batch
    pub async fn publish_batch(&mut self, events: Vec<StreamEvent>) -> Result<()> {
        for event in events {
            self.publish(event).await?;
        }
        self.flush().await
    }

    /// Flush pending messages
    pub async fn flush(&mut self) -> Result<()> {
        #[cfg(feature = "kafka")]
        {
            if let Some(ref producer) = self.producer {
                producer
                    .flush(Duration::from_secs(30))
                    .map_err(|e| anyhow::anyhow!("Failed to flush Kafka producer: {}", e))?;
                debug!("Flushed Kafka producer");
            }
        }
        #[cfg(not(feature = "kafka"))]
        {
            debug!("Mock Kafka flush");
        }
        Ok(())
    }

    /// Create a persistent consumer for a consumer group
    #[cfg(feature = "kafka")]
    pub async fn create_persistent_consumer(
        &mut self,
        group_id: &str,
        topics: Vec<String>,
    ) -> Result<ConsumerId> {
        let consumer_id = Uuid::new_v4();
        let mut client_config = ClientConfig::new();

        // Apply configuration from KafkaProducerConfig
        self.apply_client_config(&mut client_config);

        // Consumer-specific configuration
        client_config
            .set("group.id", group_id)
            .set("enable.partition.eof", "false")
            .set("session.timeout.ms", "30000")
            .set("enable.auto.commit", "false")
            .set("auto.offset.reset", "earliest");

        // Create consumer without custom context (use default)
        let consumer: StreamConsumer = client_config
            .create()
            .map_err(|e| anyhow::anyhow!("Failed to create consumer: {}", e))?;

        // Subscribe to topics
        let topic_refs: Vec<&str> = topics.iter().map(|s| s.as_str()).collect();
        consumer
            .subscribe(&topic_refs)
            .map_err(|e| anyhow::anyhow!("Failed to subscribe to topics: {}", e))?;

        let consumer_instance = ConsumerInstance {
            id: consumer_id,
            group_id: group_id.to_string(),
            consumer: Arc::new(consumer),
            is_active: Arc::new(AtomicBool::new(true)),
            message_count: Arc::new(AtomicU64::new(0)),
            error_count: Arc::new(AtomicU64::new(0)),
            last_message_time: Arc::new(RwLock::new(None)),
            stop_signal: None,
        };

        // Store the consumer instance
        let mut consumers = self.active_consumers.write().await;
        consumers.insert(consumer_id, consumer_instance);

        info!(
            "Created persistent consumer {} for group {}",
            consumer_id, group_id
        );
        Ok(consumer_id)
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn create_persistent_consumer(
        &mut self,
        group_id: &str,
        topics: Vec<String>,
    ) -> Result<ConsumerId> {
        let consumer_id = Uuid::new_v4();
        info!(
            "Mock consumer {} created for group {} with topics {:?}",
            consumer_id, group_id, topics
        );
        Ok(consumer_id)
    }

    /// Start streaming consumer with callback-based processing
    #[cfg(feature = "kafka")]
    pub async fn start_streaming_consumer(
        &self,
        consumer_id: ConsumerId,
        callback: MessageCallback,
    ) -> Result<()> {
        let consumers = self.active_consumers.read().await;
        let consumer_instance = consumers
            .get(&consumer_id)
            .ok_or_else(|| anyhow::anyhow!("Consumer {} not found", consumer_id))?;

        if !consumer_instance.is_active.load(Ordering::SeqCst) {
            return Err(anyhow::anyhow!("Consumer {} is not active", consumer_id));
        }

        let consumer = consumer_instance.consumer.clone();
        let is_active = consumer_instance.is_active.clone();
        let message_count = consumer_instance.message_count.clone();
        let error_count = consumer_instance.error_count.clone();
        let last_message_time = consumer_instance.last_message_time.clone();
        let (stop_tx, mut stop_rx) = oneshot::channel();

        // Update the consumer instance with the stop signal
        drop(consumers);
        let mut consumers = self.active_consumers.write().await;
        if let Some(instance) = consumers.get_mut(&consumer_id) {
            instance.stop_signal = Some(stop_tx);
        }
        drop(consumers);

        // Spawn the consumer loop
        let consumer_loop = async move {
            loop {
                tokio::select! {
                    _ = &mut stop_rx => {
                        info!("Consumer {} received stop signal", consumer_id);
                        break;
                    }
                    message_result = consumer.recv() => {
                        match message_result {
                            Ok(borrowed_message) => {
                                // Update last message time
                                {
                                    let mut time_guard = last_message_time.write().await;
                                    *time_guard = Some(chrono::Utc::now());
                                }

                                match Self::process_kafka_message(&borrowed_message, &callback).await {
                                    Ok(_) => {
                                        message_count.fetch_add(1, Ordering::SeqCst);
                                        // Commit the message offset
                                        if let Err(e) = consumer.commit_message(&borrowed_message, rdkafka::consumer::CommitMode::Async) {
                                            error!("Failed to commit message: {}", e);
                                            error_count.fetch_add(1, Ordering::SeqCst);
                                        }
                                    }
                                    Err(e) => {
                                        error!("Error processing message: {}", e);
                                        error_count.fetch_add(1, Ordering::SeqCst);
                                    }
                                }
                            }
                            Err(e) => {
                                error!("Error receiving message: {}", e);
                                error_count.fetch_add(1, Ordering::SeqCst);
                                tokio::time::sleep(Duration::from_millis(100)).await;
                            }
                        }
                    }
                }
            }
            is_active.store(false, Ordering::SeqCst);
        };

        tokio::spawn(consumer_loop);
        info!("Started streaming consumer {}", consumer_id);
        Ok(())
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn start_streaming_consumer(
        &self,
        consumer_id: ConsumerId,
        _callback: MessageCallback,
    ) -> Result<()> {
        info!("Mock streaming consumer {} started", consumer_id);
        Ok(())
    }

    /// Process a Kafka message and convert it to StreamEvent
    #[cfg(feature = "kafka")]
    async fn process_kafka_message(
        message: &BorrowedMessage<'_>,
        callback: &MessageCallback,
    ) -> Result<()> {
        let payload = message
            .payload()
            .ok_or_else(|| anyhow::anyhow!("Message has no payload"))?;

        let kafka_event = KafkaEvent::from_bytes(payload)?;
        let stream_event = kafka_event.to_stream_event();

        callback(stream_event)?;
        Ok(())
    }

    /// Stop a streaming consumer
    pub async fn stop_consumer(&mut self, consumer_id: ConsumerId) -> Result<()> {
        let mut consumers = self.active_consumers.write().await;
        if let Some(mut instance) = consumers.remove(&consumer_id) {
            instance.is_active.store(false, Ordering::SeqCst);
            if let Some(stop_signal) = instance.stop_signal.take() {
                let _ = stop_signal.send(());
            }
            info!("Stopped consumer {}", consumer_id);
        }
        Ok(())
    }

    /// Pause a consumer
    #[cfg(feature = "kafka")]
    pub async fn pause_consumer(&self, consumer_id: ConsumerId) -> Result<()> {
        let consumers = self.active_consumers.read().await;
        if let Some(instance) = consumers.get(&consumer_id) {
            instance
                .consumer
                .pause(&instance.consumer.assignment()?)
                .map_err(|e| anyhow::anyhow!("Failed to pause consumer: {}", e))?;
            info!("Paused consumer {}", consumer_id);
        } else {
            return Err(anyhow::anyhow!("Consumer {} not found", consumer_id));
        }
        Ok(())
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn pause_consumer(&self, consumer_id: ConsumerId) -> Result<()> {
        info!("Mock consumer {} paused", consumer_id);
        Ok(())
    }

    /// Resume a consumer
    #[cfg(feature = "kafka")]
    pub async fn resume_consumer(&self, consumer_id: ConsumerId) -> Result<()> {
        let consumers = self.active_consumers.read().await;
        if let Some(instance) = consumers.get(&consumer_id) {
            instance
                .consumer
                .resume(&instance.consumer.assignment()?)
                .map_err(|e| anyhow::anyhow!("Failed to resume consumer: {}", e))?;
            info!("Resumed consumer {}", consumer_id);
        } else {
            return Err(anyhow::anyhow!("Consumer {} not found", consumer_id));
        }
        Ok(())
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn resume_consumer(&self, consumer_id: ConsumerId) -> Result<()> {
        info!("Mock consumer {} resumed", consumer_id);
        Ok(())
    }

    /// Get consumer metrics
    pub async fn get_consumer_metrics(&self, consumer_id: ConsumerId) -> Result<ConsumerMetrics> {
        let consumers = self.active_consumers.read().await;
        if let Some(instance) = consumers.get(&consumer_id) {
            let partition_assignments = self.get_partition_assignments(&instance.consumer).await?;
            let last_message_time = *instance.last_message_time.read().await;
            Ok(ConsumerMetrics {
                consumer_id,
                group_id: instance.group_id.clone(),
                messages_processed: instance.message_count.load(Ordering::SeqCst),
                errors_encountered: instance.error_count.load(Ordering::SeqCst),
                is_active: instance.is_active.load(Ordering::SeqCst),
                last_message_time,
                partition_assignments,
            })
        } else {
            Err(anyhow::anyhow!("Consumer {} not found", consumer_id))
        }
    }

    /// Get partition assignments for a consumer
    #[cfg(feature = "kafka")]
    async fn get_partition_assignments(
        &self,
        consumer: &Arc<StreamConsumer>,
    ) -> Result<Vec<PartitionAssignment>> {
        let assignment = consumer.assignment()?;
        let mut assignments = Vec::new();

        for partition in assignment.elements() {
            let topic = partition.topic();
            let partition_id = partition.partition();
            let current_offset = partition.offset().to_raw().unwrap_or(-1);

            // Get high water mark (latest offset)
            let high_water_mark = consumer
                .fetch_watermarks(topic, partition_id, Duration::from_secs(1))
                .map(|(_, high)| high)
                .unwrap_or(-1);

            let lag = if current_offset >= 0 && high_water_mark >= 0 {
                high_water_mark - current_offset
            } else {
                0
            };

            assignments.push(PartitionAssignment {
                topic: topic.to_string(),
                partition: partition_id,
                current_offset,
                high_water_mark,
                lag,
            });
        }

        Ok(assignments)
    }

    #[cfg(not(feature = "kafka"))]
    async fn get_partition_assignments(&self, _consumer: &()) -> Result<Vec<PartitionAssignment>> {
        Ok(Vec::new())
    }

    /// Seek consumer to specific offset
    #[cfg(feature = "kafka")]
    pub async fn seek_consumer_to_offset(
        &self,
        consumer_id: ConsumerId,
        topic: &str,
        partition: i32,
        offset: i64,
    ) -> Result<()> {
        let consumers = self.active_consumers.read().await;
        if let Some(instance) = consumers.get(&consumer_id) {
            use rdkafka::{Offset, TopicPartitionList};
            let mut topic_partition_list = TopicPartitionList::new();
            topic_partition_list.add_partition_offset(topic, partition, Offset::Offset(offset))?;
            instance
                .consumer
                .seek_partitions(topic_partition_list, Duration::from_secs(10))
                .map_err(|e| anyhow::anyhow!("Failed to seek consumer: {}", e))?;
            info!(
                "Seeked consumer {} to offset {} on {}:{}",
                consumer_id, offset, topic, partition
            );
        } else {
            return Err(anyhow::anyhow!("Consumer {} not found", consumer_id));
        }
        Ok(())
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn seek_consumer_to_offset(
        &self,
        consumer_id: ConsumerId,
        topic: &str,
        partition: i32,
        offset: i64,
    ) -> Result<()> {
        info!(
            "Mock consumer {} seeked to offset {} on {}:{}",
            consumer_id, offset, topic, partition
        );
        Ok(())
    }

    /// Get list of active consumers
    pub async fn get_active_consumers(&self) -> Vec<ConsumerId> {
        let consumers = self.active_consumers.read().await;
        consumers.keys().cloned().collect()
    }

    /// Get consumer count
    pub async fn get_consumer_count(&self) -> usize {
        let consumers = self.active_consumers.read().await;
        consumers.len()
    }

    /// Consume a single event (not supported - Kafka uses streaming consumers)
    ///
    /// Note: Kafka backend uses persistent streaming consumers with callbacks.
    /// Use create_persistent_consumer() and start_streaming_consumer() instead.
    pub async fn consume(&mut self) -> Result<Option<StreamEvent>> {
        Err(anyhow::anyhow!(
            "Direct consume() not supported for Kafka backend. \
             Use create_persistent_consumer() and start_streaming_consumer() instead."
        ))
    }
}

#[async_trait]
impl StreamBackendTrait for KafkaBackend {
    fn name(&self) -> &'static str {
        "kafka"
    }

    async fn connect(&mut self) -> StreamResult<()> {
        #[cfg(feature = "kafka")]
        {
            // Initialize Kafka producer
            let mut kafka_config = ClientConfig::new();
            kafka_config
                .set("bootstrap.servers", self.kafka_config.brokers.join(","))
                .set(
                    "message.timeout.ms",
                    self.kafka_config.request_timeout_ms.to_string(),
                )
                .set(
                    "compression.type",
                    self.kafka_config.compression_type.to_string(),
                )
                .set("acks", self.kafka_config.acks.to_string())
                .set("client.id", &self.kafka_config.client_id);

            let producer: FutureProducer = kafka_config.create().map_err(|e| {
                StreamError::Connection(format!("Failed to create Kafka producer: {}", e))
            })?;

            self.producer = Some(producer);

            // Initialize admin client for topic management
            let admin_client: AdminClient<_> = kafka_config.create().map_err(|e| {
                StreamError::Connection(format!("Failed to create Kafka admin client: {}", e))
            })?;

            self.admin_client = Some(admin_client);

            info!(
                "Connected to Kafka cluster: {}",
                self.kafka_config.brokers.join(",")
            );
        }

        #[cfg(not(feature = "kafka"))]
        {
            info!("Mock Kafka connection (kafka feature not enabled)");
        }

        Ok(())
    }

    async fn disconnect(&mut self) -> StreamResult<()> {
        #[cfg(feature = "kafka")]
        {
            if let Some(producer) = self.producer.take() {
                // Flush any pending messages
                producer.flush(Duration::from_secs(5)).map_err(|e| {
                    StreamError::Connection(format!("Failed to flush Kafka producer: {}", e))
                })?;
            }
            self.admin_client = None;
            self.consumer = None;
        }

        info!("Disconnected from Kafka");
        Ok(())
    }

    async fn create_topic(&self, topic: &TopicName, partitions: u32) -> StreamResult<()> {
        #[cfg(feature = "kafka")]
        {
            if let Some(ref admin_client) = self.admin_client {
                let new_topic = NewTopic::new(
                    topic.as_str(),
                    partitions as i32,
                    TopicReplication::Fixed(1),
                );
                let opts = AdminOptions::new().operation_timeout(Some(Duration::from_secs(30)));

                admin_client
                    .create_topics(&[new_topic], &opts)
                    .await
                    .map_err(|e| {
                        StreamError::TopicCreation(format!(
                            "Failed to create topic {}: {}",
                            topic, e
                        ))
                    })?;

                info!(
                    "Created Kafka topic: {} with {} partitions",
                    topic, partitions
                );
            } else {
                return Err(StreamError::Connection(
                    "Admin client not initialized".to_string(),
                ));
            }
        }

        #[cfg(not(feature = "kafka"))]
        {
            info!(
                "Mock create topic: {} with {} partitions",
                topic, partitions
            );
        }

        Ok(())
    }

    async fn delete_topic(&self, topic: &TopicName) -> StreamResult<()> {
        #[cfg(feature = "kafka")]
        {
            if let Some(ref admin_client) = self.admin_client {
                let opts = AdminOptions::new().operation_timeout(Some(Duration::from_secs(30)));

                admin_client
                    .delete_topics(&[topic.as_str()], &opts)
                    .await
                    .map_err(|e| {
                        StreamError::TopicDeletion(format!(
                            "Failed to delete topic {}: {}",
                            topic, e
                        ))
                    })?;

                info!("Deleted Kafka topic: {}", topic);
            } else {
                return Err(StreamError::Connection(
                    "Admin client not initialized".to_string(),
                ));
            }
        }

        #[cfg(not(feature = "kafka"))]
        {
            info!("Mock delete topic: {}", topic);
        }

        Ok(())
    }

    async fn list_topics(&self) -> StreamResult<Vec<TopicName>> {
        #[cfg(feature = "kafka")]
        {
            if let Some(ref admin_client) = self.admin_client {
                let metadata = admin_client
                    .inner()
                    .fetch_metadata(None, Duration::from_secs(10))
                    .map_err(|e| {
                        StreamError::TopicList(format!("Failed to fetch metadata: {}", e))
                    })?;

                Ok(metadata
                    .topics()
                    .iter()
                    .map(|topic| topic.name().to_string().into())
                    .collect())
            } else {
                Err(StreamError::Connection(
                    "Admin client not initialized".to_string(),
                ))
            }
        }

        #[cfg(not(feature = "kafka"))]
        {
            Ok(vec!["mock-topic-1".to_string(), "mock-topic-2".to_string()])
        }
    }

    async fn send_event(&self, topic: &TopicName, event: StreamEvent) -> StreamResult<Offset> {
        #[cfg(feature = "kafka")]
        {
            if let Some(ref producer) = self.producer {
                let kafka_event = KafkaEvent::from_stream_event(event);
                let serialized = serde_json::to_string(&kafka_event)
                    .map_err(|e| StreamError::Serialization(e.to_string()))?;

                let record = FutureRecord::to(topic.as_str())
                    .key(&kafka_event.event_id)
                    .payload(&serialized);

                let result = producer
                    .send(record, Duration::from_secs(5))
                    .await
                    .map_err(|(e, _)| {
                        StreamError::Send(format!("Failed to send to Kafka: {}", e))
                    })?;

                Ok(Offset::new(result.1 as u64))
            } else {
                Err(StreamError::Connection(
                    "Producer not initialized".to_string(),
                ))
            }
        }

        #[cfg(not(feature = "kafka"))]
        {
            debug!("Mock send event to topic: {}", topic);
            Ok(Offset::new(0))
        }
    }

    async fn send_batch(
        &self,
        topic: &TopicName,
        events: Vec<StreamEvent>,
    ) -> StreamResult<Vec<Offset>> {
        let mut offsets = Vec::new();
        for event in events {
            let offset = self.send_event(topic, event).await?;
            offsets.push(offset);
        }
        Ok(offsets)
    }

    async fn receive_events(
        &self,
        topic: &TopicName,
        consumer_group: Option<&ConsumerGroup>,
        position: StreamPosition,
        max_events: usize,
    ) -> StreamResult<Vec<(StreamEvent, Offset)>> {
        #[cfg(feature = "kafka")]
        {
            use rdkafka::config::ClientConfig;
            use rdkafka::consumer::{Consumer, StreamConsumer};
            use rdkafka::message::Message;
            use rdkafka::TopicPartitionList;
            use tokio::time::{timeout, Duration};

            let group_id = consumer_group
                .map(|cg| cg.name())
                .unwrap_or("oxirs-default-group");

            // Create consumer configuration
            let mut consumer_config = ClientConfig::new();
            consumer_config
                .set("bootstrap.servers", self.kafka_config.brokers.join(","))
                .set("group.id", group_id)
                .set("enable.auto.commit", "false") // Manual commit for better control
                .set(
                    "auto.offset.reset",
                    match position {
                        StreamPosition::Beginning => "earliest",
                        StreamPosition::End => "latest",
                        StreamPosition::Offset(_) => "none", // Will seek manually
                    },
                )
                .set("session.timeout.ms", "30000")
                .set("heartbeat.interval.ms", "3000");

            // Apply security configuration if available
            if let Some(ref security_config) = self.kafka_config.security_config {
                consumer_config.set(
                    "security.protocol",
                    security_config.security_protocol.to_string(),
                );

                if let Some(ref sasl_config) = security_config.sasl_config {
                    consumer_config
                        .set("sasl.mechanism", sasl_config.mechanism.to_string())
                        .set("sasl.username", &sasl_config.username)
                        .set("sasl.password", &sasl_config.password);
                }
            }

            let consumer: StreamConsumer = consumer_config.create().map_err(|e| {
                StreamError::Connection(format!("Failed to create Kafka consumer: {}", e))
            })?;

            // Subscribe to topic
            consumer.subscribe(&[topic.as_str()]).map_err(|e| {
                StreamError::Connection(format!("Failed to subscribe to topic: {}", e))
            })?;

            // Handle specific offset positioning
            if let StreamPosition::Offset(offset_value) = position {
                let mut tpl = TopicPartitionList::new();
                tpl.add_partition_offset(
                    topic.as_str(),
                    0,
                    rdkafka::Offset::Offset(offset_value as i64),
                )
                .map_err(|e| {
                    StreamError::SeekError(format!("Failed to add partition offset: {}", e))
                })?;

                consumer
                    .seek_partitions(tpl, Duration::from_secs(10))
                    .map_err(|e| {
                        StreamError::SeekError(format!("Failed to seek to offset: {}", e))
                    })?;
            }

            let mut events = Vec::with_capacity(max_events);
            let timeout_duration = Duration::from_secs(5);

            // Consume messages up to max_events
            while events.len() < max_events {
                match timeout(timeout_duration, consumer.recv()).await {
                    Ok(Ok(message)) => {
                        if let Some(payload) = message.payload() {
                            // Deserialize the KafkaEvent
                            match serde_json::from_slice::<KafkaEvent>(payload) {
                                Ok(kafka_event) => {
                                    let event_id = kafka_event.event_id.clone();
                                    let stream_event = kafka_event.to_stream_event();
                                    let offset = Offset::new(message.offset() as u64);
                                    events.push((stream_event, offset));

                                    debug!(
                                        "Received event from Kafka: {} at offset {}",
                                        event_id,
                                        message.offset()
                                    );
                                }
                                Err(e) => {
                                    warn!("Failed to deserialize Kafka message: {}", e);
                                    continue;
                                }
                            }
                        }
                    }
                    Ok(Err(e)) => {
                        warn!("Kafka receive error: {}", e);
                        break;
                    }
                    Err(_) => {
                        // Timeout - return what we have
                        debug!("Kafka receive timeout, returning {} events", events.len());
                        break;
                    }
                }
            }

            info!(
                "Received {} events from Kafka topic: {}",
                events.len(),
                topic
            );
            Ok(events)
        }

        #[cfg(not(feature = "kafka"))]
        {
            debug!(
                "Mock receive events from topic: {}, max: {}",
                topic, max_events
            );
            Ok(vec![])
        }
    }

    async fn commit_offset(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
        partition: PartitionId,
        offset: Offset,
    ) -> StreamResult<()> {
        #[cfg(feature = "kafka")]
        {
            use rdkafka::config::ClientConfig;
            use rdkafka::consumer::{BaseConsumer, CommitMode, Consumer};
            use rdkafka::TopicPartitionList;

            // Create consumer configuration for the specific group
            let mut consumer_config = ClientConfig::new();
            consumer_config
                .set("bootstrap.servers", self.kafka_config.brokers.join(","))
                .set("group.id", consumer_group.name())
                .set("enable.auto.commit", "false")
                .set("session.timeout.ms", "30000")
                .set("heartbeat.interval.ms", "3000");

            // Apply security configuration if available
            if let Some(ref security_config) = self.kafka_config.security_config {
                consumer_config.set(
                    "security.protocol",
                    security_config.security_protocol.to_string(),
                );

                if let Some(ref sasl_config) = security_config.sasl_config {
                    consumer_config
                        .set("sasl.mechanism", sasl_config.mechanism.to_string())
                        .set("sasl.username", &sasl_config.username)
                        .set("sasl.password", &sasl_config.password);
                }
            }

            let consumer: BaseConsumer = consumer_config.create().map_err(|e| {
                StreamError::Connection(format!(
                    "Failed to create Kafka consumer for commit: {}",
                    e
                ))
            })?;

            // Create TopicPartitionList with the specific offset to commit
            let mut tpl = TopicPartitionList::new();
            tpl.add_partition_offset(
                topic.as_str(),
                partition.value() as i32,
                rdkafka::Offset::Offset(offset.value() as i64 + 1), // Kafka commits the next offset
            )
            .map_err(|e| {
                StreamError::CommitError(format!(
                    "Failed to add partition offset for commit: {}",
                    e
                ))
            })?;

            // Commit the offset
            consumer
                .commit(&tpl, CommitMode::Sync)
                .map_err(|e| StreamError::CommitError(format!("Failed to commit offset: {}", e)))?;

            debug!(
                "Committed offset {} for topic: {}, partition: {}, group: {}",
                offset.value(),
                topic,
                partition.value(),
                consumer_group.name()
            );
            Ok(())
        }

        #[cfg(not(feature = "kafka"))]
        {
            debug!(
                "Mock commit offset for topic: {}, partition: {}, offset: {}",
                topic,
                partition.value(),
                offset.value()
            );
            Ok(())
        }
    }

    async fn seek(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
        partition: PartitionId,
        position: StreamPosition,
    ) -> StreamResult<()> {
        #[cfg(feature = "kafka")]
        {
            use rdkafka::config::ClientConfig;
            use rdkafka::consumer::{BaseConsumer, Consumer};
            use rdkafka::TopicPartitionList;
            use tokio::time::Duration;

            // Create consumer configuration for the specific group
            let mut consumer_config = ClientConfig::new();
            consumer_config
                .set("bootstrap.servers", self.kafka_config.brokers.join(","))
                .set("group.id", consumer_group.name())
                .set("enable.auto.commit", "false")
                .set("session.timeout.ms", "30000")
                .set("heartbeat.interval.ms", "3000");

            // Apply security configuration if available
            if let Some(ref security_config) = self.kafka_config.security_config {
                consumer_config.set(
                    "security.protocol",
                    security_config.security_protocol.to_string(),
                );

                if let Some(ref sasl_config) = security_config.sasl_config {
                    consumer_config
                        .set("sasl.mechanism", sasl_config.mechanism.to_string())
                        .set("sasl.username", &sasl_config.username)
                        .set("sasl.password", &sasl_config.password);
                }
            }

            let consumer: BaseConsumer = consumer_config.create().map_err(|e| {
                StreamError::Connection(format!("Failed to create Kafka consumer for seek: {}", e))
            })?;

            // Create TopicPartitionList with the position to seek to
            let mut tpl = TopicPartitionList::new();

            let kafka_offset = match position {
                StreamPosition::Beginning => rdkafka::Offset::Beginning,
                StreamPosition::End => rdkafka::Offset::End,
                StreamPosition::Offset(offset_value) => {
                    rdkafka::Offset::Offset(offset_value as i64)
                }
            };

            tpl.add_partition_offset(topic.as_str(), partition.value() as i32, kafka_offset)
                .map_err(|e| {
                    StreamError::SeekError(format!("Failed to add partition for seek: {}", e))
                })?;

            // Perform the seek operation
            consumer
                .seek_partitions(tpl, Duration::from_secs(10))
                .map_err(|e| StreamError::SeekError(format!("Failed to seek: {}", e)))?;

            info!(
                "Seeked to position {:?} for topic: {}, partition: {}, group: {}",
                position,
                topic,
                partition.value(),
                consumer_group.name()
            );
            Ok(())
        }

        #[cfg(not(feature = "kafka"))]
        {
            debug!(
                "Mock seek for topic: {}, partition: {} to position: {:?}",
                topic,
                partition.value(),
                position
            );
            Ok(())
        }
    }

    async fn get_consumer_lag(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
    ) -> StreamResult<HashMap<PartitionId, u64>> {
        #[cfg(feature = "kafka")]
        {
            use rdkafka::config::ClientConfig;
            use rdkafka::consumer::{BaseConsumer, Consumer};
            use rdkafka::TopicPartitionList;
            use std::collections::HashMap;

            // Create consumer configuration for the specific group
            let mut consumer_config = ClientConfig::new();
            consumer_config
                .set("bootstrap.servers", self.kafka_config.brokers.join(","))
                .set("group.id", consumer_group.name())
                .set("enable.auto.commit", "false")
                .set("session.timeout.ms", "30000")
                .set("heartbeat.interval.ms", "3000");

            // Apply security configuration if available
            if let Some(ref security_config) = self.kafka_config.security_config {
                consumer_config.set(
                    "security.protocol",
                    security_config.security_protocol.to_string(),
                );

                if let Some(ref sasl_config) = security_config.sasl_config {
                    consumer_config
                        .set("sasl.mechanism", sasl_config.mechanism.to_string())
                        .set("sasl.username", &sasl_config.username)
                        .set("sasl.password", &sasl_config.password);
                }
            }

            let consumer: BaseConsumer = consumer_config.create().map_err(|e| {
                StreamError::Connection(format!(
                    "Failed to create Kafka consumer for lag check: {}",
                    e
                ))
            })?;

            // Get topic metadata to find all partitions
            let metadata = consumer
                .fetch_metadata(Some(topic.as_str()), Duration::from_secs(10))
                .map_err(|e| {
                    StreamError::TopicMetadata(format!("Failed to fetch topic metadata: {}", e))
                })?;

            let mut lag_map = HashMap::new();

            if let Some(topic_metadata) = metadata.topics().first() {
                for partition_metadata in topic_metadata.partitions() {
                    let partition_id = PartitionId::new(partition_metadata.id() as u32);

                    // Get high water mark (latest offset)
                    let mut tpl = TopicPartitionList::new();
                    tpl.add_partition_offset(
                        topic.as_str(),
                        partition_metadata.id(),
                        rdkafka::Offset::End,
                    )
                    .map_err(|e| {
                        StreamError::TopicMetadata(format!(
                            "Failed to add partition for high water mark: {}",
                            e
                        ))
                    })?;

                    let high_water_marks = consumer
                        .committed_offsets(tpl, Duration::from_secs(10))
                        .map_err(|e| {
                            StreamError::TopicMetadata(format!(
                                "Failed to get high water marks: {}",
                                e
                            ))
                        })?;

                    // Get consumer group's committed offset
                    let mut committed_tpl = TopicPartitionList::new();
                    committed_tpl.add_partition(topic.as_str(), partition_metadata.id());

                    let committed_offsets = consumer
                        .committed_offsets(committed_tpl, Duration::from_secs(10))
                        .map_err(|e| {
                            StreamError::TopicMetadata(format!(
                                "Failed to get committed offsets: {}",
                                e
                            ))
                        })?;

                    // Calculate lag
                    if let Some(high_water_element) = high_water_marks.elements().first() {
                        if let Some(committed_element) = committed_offsets.elements().first() {
                            let high_water_offset = match high_water_element.offset() {
                                rdkafka::Offset::Offset(offset) => offset as u64,
                                rdkafka::Offset::End => 0, // Topic is empty
                                _ => 0,
                            };

                            let committed_offset = match committed_element.offset() {
                                rdkafka::Offset::Offset(offset) => offset as u64,
                                rdkafka::Offset::Invalid => 0, // No committed offset yet
                                _ => 0,
                            };

                            let lag = high_water_offset.saturating_sub(committed_offset);

                            lag_map.insert(partition_id, lag);
                            debug!(
                                "Partition {}: lag = {} (high water: {}, committed: {})",
                                partition_metadata.id(),
                                lag,
                                high_water_offset,
                                committed_offset
                            );
                        }
                    }
                }
            }

            info!(
                "Retrieved consumer lag for topic: {} with {} partitions",
                topic,
                lag_map.len()
            );
            Ok(lag_map)
        }

        #[cfg(not(feature = "kafka"))]
        {
            debug!("Mock get consumer lag for topic: {}", topic);
            let mut lag_map = HashMap::new();
            lag_map.insert(PartitionId::new(0), 0);
            Ok(lag_map)
        }
    }

    async fn get_topic_metadata(&self, topic: &TopicName) -> StreamResult<HashMap<String, String>> {
        #[cfg(feature = "kafka")]
        {
            let mut metadata = HashMap::new();
            metadata.insert("backend".to_string(), "kafka".to_string());
            metadata.insert("topic".to_string(), topic.to_string());
            metadata.insert("brokers".to_string(), self.kafka_config.brokers.join(","));
            Ok(metadata)
        }

        #[cfg(not(feature = "kafka"))]
        {
            let mut metadata = HashMap::new();
            metadata.insert("backend".to_string(), "kafka".to_string());
            metadata.insert("topic".to_string(), topic.to_string());
            metadata.insert("brokers".to_string(), "mock-broker:9092".to_string());
            Ok(metadata)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_kafka_backend_creation() {
        let mut config = StreamConfig::default();
        config.backend = crate::StreamBackendType::Kafka {
            brokers: vec!["localhost:9092".to_string()],
            security_protocol: None,
            sasl_config: None,
        };
        config.topic = "test_topic".to_string();
        config.batch_size = 100;

        let backend = KafkaBackend::new(config);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_kafka_event_conversion() {
        use crate::EventMetadata;
        use chrono::Utc;

        let metadata = EventMetadata {
            event_id: "test".to_string(),
            timestamp: Utc::now(),
            source: "test".to_string(),
            user: None,
            context: None,
            caused_by: None,
            version: "1.0".to_string(),
            properties: HashMap::new(),
            checksum: None,
        };

        let stream_event = StreamEvent::TripleAdded {
            subject: "test:subject".to_string(),
            predicate: "test:predicate".to_string(),
            object: "test:object".to_string(),
            graph: Some("test:graph".to_string()),
            metadata,
        };

        let kafka_event = KafkaEvent::from(stream_event);
        assert_eq!(kafka_event.event_type, "triple_added");
        assert!(kafka_event.partition_key.is_some());
    }

    #[tokio::test]
    async fn test_stats_update() {
        let mut config = StreamConfig::default();
        config.backend = crate::StreamBackendType::Kafka {
            brokers: vec!["localhost:9092".to_string()],
            security_protocol: None,
            sasl_config: None,
        };
        config.topic = "test_topic".to_string();
        config.batch_size = 100;

        let backend = KafkaBackend::new(config).unwrap();

        // Test stats update
        backend.update_stats(1024, 50, false).await;
        let stats = backend.get_stats().await;

        assert_eq!(stats.events_published, 1);
        assert_eq!(stats.bytes_sent, 1024);
        assert_eq!(stats.max_latency_ms, 50);
        assert_eq!(stats.avg_latency_ms, 50.0);
    }
}
