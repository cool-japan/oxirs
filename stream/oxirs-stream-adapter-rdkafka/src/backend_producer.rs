//! Kafka backend core and producer-side operations.
//!
//! Defines the [`KafkaBackend`] struct together with connection setup,
//! client configuration, topic creation, single/batch event publishing,
//! flushing, and producer statistics tracking.

use super::backend_types::ConsumerInstance;
use super::{KafkaEvent, KafkaProducerConfig, KafkaProducerStats};
use crate::backend_types::ConsumerId;
use anyhow::Result;
use oxirs_stream::{StreamConfig, StreamEvent};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, error, info};

#[cfg(feature = "kafka")]
use rdkafka::producer::Producer;
#[cfg(feature = "kafka")]
use std::time::Duration;

#[cfg(feature = "kafka")]
use rdkafka::{
    admin::{AdminClient, AdminOptions, NewTopic, TopicReplication},
    config::ClientConfig,
    consumer::StreamConsumer,
    producer::{FutureProducer, FutureRecord},
};

/// Kafka backend implementation that coordinates all modules
pub struct KafkaBackend {
    pub(crate) config: StreamConfig,
    pub(crate) kafka_config: KafkaProducerConfig,
    #[cfg(feature = "kafka")]
    pub(crate) producer: Option<FutureProducer>,
    #[cfg(feature = "kafka")]
    pub(crate) consumer: Option<StreamConsumer>,
    #[cfg(feature = "kafka")]
    pub(crate) admin_client: Option<AdminClient<rdkafka::client::DefaultClientContext>>,
    #[cfg(not(feature = "kafka"))]
    pub(crate) _phantom: std::marker::PhantomData<()>,
    pub(crate) stats: Arc<RwLock<KafkaProducerStats>>,
    // Consumer management
    pub(crate) active_consumers: Arc<RwLock<HashMap<ConsumerId, ConsumerInstance>>>,
}

impl KafkaBackend {
    /// Create a new Kafka backend with the given configuration
    pub fn new(config: StreamConfig) -> Result<Self> {
        let kafka_config =
            if let oxirs_stream::StreamBackendType::Kafka { brokers, .. } = &config.backend {
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
    pub(crate) fn apply_client_config(&self, client_config: &mut ClientConfig) {
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
        tracing::warn!("Kafka feature not enabled, using mock backend");
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
        let _start_time = Instant::now();

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
                        let latency_ms = _start_time.elapsed().as_millis() as u64;
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
    pub(crate) async fn update_stats(&self, bytes_sent: usize, latency_ms: u64, is_error: bool) {
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
}
