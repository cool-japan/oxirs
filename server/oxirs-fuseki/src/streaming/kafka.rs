//! Apache Kafka integration for event streaming

use std::{collections::HashMap, sync::Arc, time::Duration};
use async_trait::async_trait;
use tokio::sync::{Mutex, RwLock};
use rdkafka::{
    config::ClientConfig,
    consumer::{Consumer, StreamConsumer, ConsumerContext, Rebalance},
    producer::{FutureProducer, FutureRecord},
    message::{Message, Headers, OwnedHeaders},
    ClientContext, Offset, TopicPartitionList,
};
use serde_json;

use crate::{
    error::{Error, Result},
    streaming::{KafkaConfig, RDFEvent, StreamProducer, StreamConsumer, EventHandler},
};

/// Kafka producer for sending RDF events
pub struct KafkaProducer {
    producer: FutureProducer,
    config: KafkaConfig,
    topic_cache: Arc<RwLock<HashMap<String, bool>>>,
}

impl KafkaProducer {
    /// Create a new Kafka producer
    pub async fn new(config: KafkaConfig) -> Result<Self> {
        let mut client_config = ClientConfig::new();
        
        // Set bootstrap servers
        let brokers = config.brokers.join(",");
        client_config.set("bootstrap.servers", &brokers);
        
        // Producer configuration
        client_config
            .set("compression.type", &config.producer.compression)
            .set("batch.size", &config.producer.batch_size.to_string())
            .set("linger.ms", &config.producer.linger_ms.to_string())
            .set("request.timeout.ms", &config.producer.request_timeout_ms.to_string())
            .set("acks", "all")
            .set("enable.idempotence", "true");
        
        // Enable transactions if configured
        if config.enable_transactions {
            client_config.set("transactional.id", "oxirs-producer");
        }
        
        let producer: FutureProducer = client_config
            .create()
            .map_err(|e| Error::Custom(format!("Failed to create Kafka producer: {}", e)))?;
        
        // Initialize transactional producer if enabled
        if config.enable_transactions {
            producer.init_transactions(Duration::from_secs(30))
                .map_err(|e| Error::Custom(format!("Failed to init transactions: {}", e)))?;
        }
        
        Ok(Self {
            producer,
            config,
            topic_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Get topic name for an event type
    fn get_topic_name(&self, event: &RDFEvent) -> String {
        let event_type = match event {
            RDFEvent::TripleAdded { .. } => "triple.added",
            RDFEvent::TripleRemoved { .. } => "triple.removed",
            RDFEvent::QuadAdded { .. } => "quad.added",
            RDFEvent::QuadRemoved { .. } => "quad.removed",
            RDFEvent::GraphCleared { .. } => "graph.cleared",
            RDFEvent::Transaction { .. } => "transaction",
        };
        
        format!("{}.{}", self.config.topic_prefix, event_type)
    }

    /// Create message headers
    fn create_headers(event: &RDFEvent) -> OwnedHeaders {
        let mut headers = OwnedHeaders::new();
        
        headers = headers.insert(rdkafka::message::Header {
            key: "event_type",
            value: Some(match event {
                RDFEvent::TripleAdded { .. } => b"triple_added",
                RDFEvent::TripleRemoved { .. } => b"triple_removed",
                RDFEvent::QuadAdded { .. } => b"quad_added",
                RDFEvent::QuadRemoved { .. } => b"quad_removed",
                RDFEvent::GraphCleared { .. } => b"graph_cleared",
                RDFEvent::Transaction { .. } => b"transaction",
            }),
        });
        
        // Add timestamp header
        if let Ok(timestamp) = match event {
            RDFEvent::TripleAdded { timestamp, .. } |
            RDFEvent::TripleRemoved { timestamp, .. } |
            RDFEvent::QuadAdded { timestamp, .. } |
            RDFEvent::QuadRemoved { timestamp, .. } |
            RDFEvent::GraphCleared { timestamp, .. } |
            RDFEvent::Transaction { timestamp, .. } => Ok(timestamp),
        } {
            headers = headers.insert(rdkafka::message::Header {
                key: "timestamp",
                value: Some(timestamp.to_string().as_bytes()),
            });
        }
        
        headers
    }
}

#[async_trait]
impl StreamProducer for KafkaProducer {
    async fn send(&self, event: RDFEvent) -> Result<()> {
        let topic = self.get_topic_name(&event);
        let payload = serde_json::to_vec(&event)
            .map_err(|e| Error::Custom(format!("Failed to serialize event: {}", e)))?;
        
        let headers = Self::create_headers(&event);
        
        // Create the record
        let record = FutureRecord::to(&topic)
            .payload(&payload)
            .headers(headers);
        
        // Send the record
        let delivery_result = self.producer
            .send(record, Duration::from_secs(10))
            .await;
        
        match delivery_result {
            Ok((partition, offset)) => {
                tracing::debug!(
                    "Event sent to topic {} partition {} offset {}",
                    topic, partition, offset
                );
                Ok(())
            }
            Err((e, _)) => {
                Err(Error::Custom(format!("Failed to send event: {}", e)))
            }
        }
    }

    async fn send_batch(&self, events: Vec<RDFEvent>) -> Result<()> {
        if self.config.enable_transactions {
            // Begin transaction
            self.producer.begin_transaction()
                .map_err(|e| Error::Custom(format!("Failed to begin transaction: {}", e)))?;
        }
        
        // Send all events
        for event in events {
            if let Err(e) = self.send(event).await {
                if self.config.enable_transactions {
                    // Abort transaction on error
                    let _ = self.producer.abort_transaction(Duration::from_secs(10));
                }
                return Err(e);
            }
        }
        
        if self.config.enable_transactions {
            // Commit transaction
            self.producer.commit_transaction(Duration::from_secs(30))
                .map_err(|e| Error::Custom(format!("Failed to commit transaction: {}", e)))?;
        }
        
        Ok(())
    }

    async fn flush(&self) -> Result<()> {
        self.producer.flush(Duration::from_secs(30))
            .map_err(|e| Error::Custom(format!("Failed to flush producer: {}", e)))?;
        Ok(())
    }
}

/// Kafka consumer context
struct CustomContext;

impl ClientContext for CustomContext {}

impl ConsumerContext for CustomContext {
    fn pre_rebalance(&self, rebalance: &Rebalance) {
        match rebalance {
            Rebalance::Assign(partitions) => {
                tracing::info!("Partitions assigned: {:?}", partitions);
            }
            Rebalance::Revoke => {
                tracing::info!("Partitions revoked");
            }
            Rebalance::Error(e) => {
                tracing::error!("Rebalance error: {}", e);
            }
        }
    }
}

/// Kafka consumer for receiving RDF events
pub struct KafkaConsumer {
    consumer: Arc<StreamConsumer<CustomContext>>,
    config: KafkaConfig,
    handler: Arc<Mutex<Option<Box<dyn EventHandler>>>>,
    running: Arc<RwLock<bool>>,
}

impl KafkaConsumer {
    /// Create a new Kafka consumer
    pub async fn new(config: KafkaConfig) -> Result<Self> {
        let context = CustomContext;
        let mut client_config = ClientConfig::new();
        
        // Set bootstrap servers
        let brokers = config.brokers.join(",");
        client_config.set("bootstrap.servers", &brokers);
        
        // Consumer configuration
        client_config
            .set("group.id", &config.consumer.group_id)
            .set("enable.auto.commit", &config.consumer.enable_auto_commit.to_string())
            .set("auto.offset.reset", &config.consumer.auto_offset_reset)
            .set("max.poll.records", &config.consumer.max_poll_records.to_string())
            .set("session.timeout.ms", "30000")
            .set("heartbeat.interval.ms", "3000");
        
        let consumer: StreamConsumer<CustomContext> = client_config
            .create_with_context(context)
            .map_err(|e| Error::Custom(format!("Failed to create Kafka consumer: {}", e)))?;
        
        Ok(Self {
            consumer: Arc::new(consumer),
            config,
            handler: Arc::new(Mutex::new(None)),
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start consuming messages
    async fn start_consuming(&self) {
        let consumer = self.consumer.clone();
        let handler = self.handler.clone();
        let running = self.running.clone();
        
        // Subscribe to topics
        let topics: Vec<String> = vec![
            format!("{}.triple.added", self.config.topic_prefix),
            format!("{}.triple.removed", self.config.topic_prefix),
            format!("{}.quad.added", self.config.topic_prefix),
            format!("{}.quad.removed", self.config.topic_prefix),
            format!("{}.graph.cleared", self.config.topic_prefix),
            format!("{}.transaction", self.config.topic_prefix),
        ];
        
        let topic_refs: Vec<&str> = topics.iter().map(|s| s.as_str()).collect();
        
        if let Err(e) = consumer.subscribe(&topic_refs) {
            tracing::error!("Failed to subscribe to topics: {}", e);
            return;
        }
        
        tokio::spawn(async move {
            while *running.read().await {
                match consumer.recv().await {
                    Ok(message) => {
                        if let Some(payload) = message.payload() {
                            match serde_json::from_slice::<RDFEvent>(payload) {
                                Ok(event) => {
                                    if let Some(handler) = &*handler.lock().await {
                                        if let Err(e) = handler.handle(event).await {
                                            handler.on_error(Box::new(e)).await;
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::error!("Failed to deserialize event: {}", e);
                                }
                            }
                        }
                        
                        // Commit offset if auto-commit is disabled
                        if !consumer.context().client().config().get("enable.auto.commit").unwrap_or("true".to_string()).parse::<bool>().unwrap_or(true) {
                            if let Err(e) = consumer.commit_message(&message, rdkafka::consumer::CommitMode::Async) {
                                tracing::error!("Failed to commit message: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Kafka consumer error: {}", e);
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
            }
        });
    }
}

#[async_trait]
impl StreamConsumer for KafkaConsumer {
    async fn subscribe(&self, handler: Box<dyn EventHandler>) -> Result<()> {
        *self.handler.lock().await = Some(handler);
        *self.running.write().await = true;
        self.start_consuming().await;
        Ok(())
    }

    async fn unsubscribe(&self) -> Result<()> {
        *self.running.write().await = false;
        self.consumer.unsubscribe();
        Ok(())
    }

    async fn commit(&self) -> Result<()> {
        self.consumer
            .commit_consumer_state(rdkafka::consumer::CommitMode::Sync)
            .map_err(|e| Error::Custom(format!("Failed to commit offsets: {}", e)))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topic_naming() {
        let config = KafkaConfig {
            brokers: vec!["localhost:9092".to_string()],
            topic_prefix: "oxirs".to_string(),
            producer: Default::default(),
            consumer: Default::default(),
            enable_transactions: false,
        };
        
        let producer = KafkaProducer {
            producer: unsafe { std::mem::zeroed() }, // Don't actually use this
            config,
            topic_cache: Arc::new(RwLock::new(HashMap::new())),
        };
        
        let event = RDFEvent::GraphCleared {
            graph: "test".to_string(),
            timestamp: 12345,
        };
        
        assert_eq!(producer.get_topic_name(&event), "oxirs.graph.cleared");
    }
}