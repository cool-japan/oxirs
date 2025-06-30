//! Kafka Streaming Backend - Modular Implementation
//!
//! This module provides a modular Kafka backend implementation for streaming RDF data.
//! The implementation is broken down into focused modules for better maintainability
//! and adherence to the 2000-line file policy.

pub mod config;
pub mod message;

// Re-export public types
pub use config::*;
pub use message::*;

use crate::backend::{StreamBackend as StreamBackendTrait, StreamBackendConfig};
use crate::error::{StreamError, StreamResult};
use crate::{StreamBackend, StreamConfig, StreamEvent};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

#[cfg(feature = "kafka")]
use rdkafka::{
    config::ClientConfig,
    producer::{FutureProducer, FutureRecord},
    consumer::{StreamConsumer, Consumer},
    admin::{AdminClient, NewTopic},
};

/// Kafka backend implementation
pub struct KafkaBackend {
    config: StreamConfig,
    kafka_config: KafkaProducerConfig,
    #[cfg(feature = "kafka")]
    producer: Option<FutureProducer>,
    #[cfg(feature = "kafka")]
    admin_client: Option<AdminClient<rdkafka::client::DefaultClientContext>>,
    #[cfg(not(feature = "kafka"))]
    _phantom: std::marker::PhantomData<()>,
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

impl KafkaBackend {
    pub fn new(config: StreamConfig) -> Result<Self> {
        let kafka_config = if let StreamBackend::Kafka { brokers, .. } = &config.backend {
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
            admin_client: None,
            #[cfg(not(feature = "kafka"))]
            _phantom: std::marker::PhantomData,
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
            .set("acks", &self.kafka_config.acks.to_string())
            .set("retries", &self.kafka_config.retries.to_string())
            .set("batch.size", &self.kafka_config.batch_size.to_string())
            .set("linger.ms", &self.kafka_config.linger_ms.to_string())
            .set("buffer.memory", &self.kafka_config.buffer_memory.to_string())
            .set("compression.type", &self.kafka_config.compression_type.to_string())
            .set("max.in.flight.requests.per.connection", &self.kafka_config.max_in_flight_requests.to_string())
            .set("request.timeout.ms", &self.kafka_config.request_timeout_ms.to_string())
            .set("delivery.timeout.ms", &self.kafka_config.delivery_timeout_ms.to_string());

        // Enable idempotence for exactly-once semantics
        if self.kafka_config.enable_idempotence {
            client_config.set("enable.idempotence", "true");
        }

        // Transaction ID for transactional producer
        if let Some(ref transaction_id) = self.kafka_config.transaction_id {
            client_config.set("transactional.id", transaction_id);
        }

        // Security configuration
        if let Some(ref security_config) = self.kafka_config.security_config {
            client_config.set("security.protocol", &security_config.security_protocol.to_string());
            
            if let Some(ref sasl_config) = security_config.sasl_config {
                client_config
                    .set("sasl.mechanism", &sasl_config.mechanism.to_string())
                    .set("sasl.username", &sasl_config.username)
                    .set("sasl.password", &sasl_config.password);
            }
        }

        // Create producer
        let producer: FutureProducer = client_config.create()
            .map_err(|e| anyhow::anyhow!("Failed to create Kafka producer: {}", e))?;

        // Create admin client
        let admin_client: AdminClient<rdkafka::client::DefaultClientContext> = client_config.create()
            .map_err(|e| anyhow::anyhow!("Failed to create Kafka admin client: {}", e))?;

        self.producer = Some(producer);
        self.admin_client = Some(admin_client);

        info!("Connected to Kafka cluster at {}", self.kafka_config.brokers.join(","));
        Ok(())
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn connect(&mut self) -> Result<()> {
        warn!("Kafka feature not enabled, using mock backend");
        Ok(())
    }

    #[cfg(feature = "kafka")]
    pub async fn create_topic(&self, topic_name: &str, partitions: i32, replication_factor: i16) -> Result<()> {
        if let Some(ref admin_client) = self.admin_client {
            use rdkafka::admin::{AdminOptions, TopicReplication};

            let new_topic = NewTopic::new(topic_name, partitions, TopicReplication::Fixed(replication_factor));
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
                                    return Err(anyhow::anyhow!("Failed to create topic: {}", error));
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

    pub async fn publish(&mut self, event: StreamEvent) -> Result<()> {
        let kafka_event = KafkaEvent::from(event);
        let topic_name = kafka_event.get_topic_name("rdf");

        #[cfg(feature = "kafka")]
        {
            if self.producer.is_none() {
                self.connect().await?;
            }

            if let Some(ref producer) = self.producer {
                let payload = kafka_event.to_bytes()
                    .map_err(|e| anyhow::anyhow!("Failed to serialize event: {}", e))?;

                let mut record = FutureRecord::to(&topic_name)
                    .payload(&payload)
                    .key(&kafka_event.event_id);

                // Set partition key if available
                if let Some(ref partition_key) = kafka_event.partition_key {
                    record = record.key(partition_key);
                }

                match producer.send(record, Duration::from_secs(30)).await {
                    Ok(delivery) => {
                        debug!("Published event to Kafka: {} to topic {}", kafka_event.event_id, topic_name);
                    }
                    Err((kafka_error, _)) => {
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
            debug!("Mock Kafka publish: {} to topic {}", kafka_event.event_id, topic_name);
        }

        Ok(())
    }

    pub async fn publish_batch(&mut self, events: Vec<StreamEvent>) -> Result<()> {
        for event in events {
            self.publish(event).await?;
        }
        self.flush().await
    }

    pub async fn flush(&mut self) -> Result<()> {
        #[cfg(feature = "kafka")]
        {
            if let Some(ref producer) = self.producer {
                producer.flush(Duration::from_secs(30))
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

#[async_trait]
impl StreamBackendTrait for KafkaBackend {
    async fn initialize(&mut self) -> StreamResult<()> {
        self.connect().await.map_err(|e| StreamError::Configuration(e.to_string()))?;
        Ok(())
    }

    async fn publish_event(&mut self, event: StreamEvent) -> StreamResult<()> {
        self.publish(event).await.map_err(|e| StreamError::PublishError(e.to_string()))
    }

    async fn flush(&mut self) -> StreamResult<()> {
        self.flush().await.map_err(|e| StreamError::FlushError(e.to_string()))
    }

    async fn close(&mut self) -> StreamResult<()> {
        self.flush().await.map_err(|e| StreamError::FlushError(e.to_string()))?;
        info!("Kafka backend closed");
        Ok(())
    }

    fn get_config(&self) -> StreamBackendConfig {
        StreamBackendConfig {
            backend_type: "kafka".to_string(),
            url: self.kafka_config.brokers.join(","),
            batch_size: Some(self.kafka_config.batch_size as usize),
            buffer_size: Some(self.kafka_config.buffer_memory as usize),
            timeout: Some(Duration::from_millis(self.kafka_config.request_timeout_ms as u64)),
            additional_config: {
                let mut config = HashMap::new();
                config.insert("client_id".to_string(), self.kafka_config.client_id.clone());
                config.insert("compression_type".to_string(), self.kafka_config.compression_type.to_string());
                config.insert("acks".to_string(), self.kafka_config.acks.to_string());
                config
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_kafka_backend_creation() {
        let config = StreamConfig {
            backend: StreamBackend::Kafka {
                brokers: vec!["localhost:9092".to_string()],
                topic: "test_topic".to_string(),
            },
            batch_size: 100,
            flush_interval: Duration::from_millis(100),
            max_retries: 3,
            timeout: Duration::from_secs(30),
        };

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
}