//! # Kafka Streaming Backend
//!
//! Apache Kafka support for streaming RDF data.
//! 
//! This module provides high-performance Kafka integration for streaming
//! RDF updates, patches, and SPARQL operations in real-time.

use anyhow::{anyhow, Result};
use crate::{StreamEvent, StreamConfig, StreamBackend, RdfPatch, PatchOperation, EventMetadata};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tokio::time;
use tracing::{debug, error, info, warn};

#[cfg(feature = "kafka")]
use rdkafka::{
    admin::{AdminClient, AdminOptions, NewTopic, TopicReplication},
    client::DefaultClientContext,
    config::ClientConfig,
    consumer::{Consumer, StreamConsumer as KafkaStreamConsumer},
    producer::{Producer, FutureProducer, FutureRecord},
    Message, TopicPartitionList,
};

/// Serializable event for Kafka transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaEvent {
    pub event_type: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub data: serde_json::Value,
    pub source: Option<String>,
    pub correlation_id: String,
}

impl From<StreamEvent> for KafkaEvent {
    fn from(event: StreamEvent) -> Self {
        let (event_type, data) = match event {
            StreamEvent::TripleAdded { subject, predicate, object } => {
                ("triple_added".to_string(), serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object
                }))
            }
            StreamEvent::TripleRemoved { subject, predicate, object } => {
                ("triple_removed".to_string(), serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object
                }))
            }
            StreamEvent::GraphCleared { graph } => {
                ("graph_cleared".to_string(), serde_json::json!({
                    "graph": graph
                }))
            }
            StreamEvent::SparqlUpdate { query } => {
                ("sparql_update".to_string(), serde_json::json!({
                    "query": query
                }))
            }
        };
        
        Self {
            event_type,
            timestamp: chrono::Utc::now(),
            data,
            source: None,
            correlation_id: uuid::Uuid::new_v4().to_string(),
        }
    }
}

impl TryFrom<KafkaEvent> for StreamEvent {
    type Error = anyhow::Error;
    
    fn try_from(kafka_event: KafkaEvent) -> Result<Self> {
        match kafka_event.event_type.as_str() {
            "triple_added" => {
                let subject = kafka_event.data["subject"].as_str()
                    .ok_or_else(|| anyhow!("Missing subject"))?.to_string();
                let predicate = kafka_event.data["predicate"].as_str()
                    .ok_or_else(|| anyhow!("Missing predicate"))?.to_string();
                let object = kafka_event.data["object"].as_str()
                    .ok_or_else(|| anyhow!("Missing object"))?.to_string();
                Ok(StreamEvent::TripleAdded { subject, predicate, object })
            }
            "triple_removed" => {
                let subject = kafka_event.data["subject"].as_str()
                    .ok_or_else(|| anyhow!("Missing subject"))?.to_string();
                let predicate = kafka_event.data["predicate"].as_str()
                    .ok_or_else(|| anyhow!("Missing predicate"))?.to_string();
                let object = kafka_event.data["object"].as_str()
                    .ok_or_else(|| anyhow!("Missing object"))?.to_string();
                Ok(StreamEvent::TripleRemoved { subject, predicate, object })
            }
            "graph_cleared" => {
                let graph = kafka_event.data["graph"].as_str().map(|s| s.to_string());
                Ok(StreamEvent::GraphCleared { graph })
            }
            "sparql_update" => {
                let query = kafka_event.data["query"].as_str()
                    .ok_or_else(|| anyhow!("Missing query"))?.to_string();
                Ok(StreamEvent::SparqlUpdate { query })
            }
            _ => Err(anyhow!("Unknown event type: {}", kafka_event.event_type))
        }
    }
}

/// Kafka producer for RDF streaming
pub struct KafkaProducer {
    config: StreamConfig,
    #[cfg(feature = "kafka")]
    producer: Option<FutureProducer>,
    #[cfg(not(feature = "kafka"))]
    _phantom: std::marker::PhantomData<()>,
    pending_events: Vec<KafkaEvent>,
    stats: ProducerStats,
}

#[derive(Debug, Default)]
struct ProducerStats {
    events_published: u64,
    events_failed: u64,
    bytes_sent: u64,
    last_flush: Option<chrono::DateTime<chrono::Utc>>,
}

impl KafkaProducer {
    pub fn new(config: StreamConfig) -> Result<Self> {
        #[cfg(feature = "kafka")]
        {
            let producer = if let StreamBackend::Kafka { brokers } = &config.backend {
                let mut client_config = ClientConfig::new();
                client_config
                    .set("bootstrap.servers", brokers.join(","))
                    .set("message.timeout.ms", "30000")
                    .set("queue.buffering.max.messages", "100000")
                    .set("queue.buffering.max.kbytes", "1048576")
                    .set("batch.size", "65536")
                    .set("linger.ms", "100")
                    .set("compression.type", "snappy")
                    .set("acks", "all")
                    .set("retries", "10")
                    .set("retry.backoff.ms", "100")
                    .set("enable.idempotence", "true");
                
                let producer: FutureProducer = client_config.create()
                    .map_err(|e| anyhow!("Failed to create Kafka producer: {}", e))?;
                
                Some(producer)
            } else {
                None
            };
            
            Ok(Self {
                config,
                producer,
                pending_events: Vec::new(),
                stats: ProducerStats::default(),
            })
        }
        #[cfg(not(feature = "kafka"))]
        {
            warn!("Kafka feature not enabled, using mock producer");
            Ok(Self {
                config,
                _phantom: std::marker::PhantomData,
                pending_events: Vec::new(),
                stats: ProducerStats::default(),
            })
        }
    }
    
    /// Create topic if it doesn't exist
    #[cfg(feature = "kafka")]
    pub async fn ensure_topic(&self) -> Result<()> {
        if let StreamBackend::Kafka { brokers, .. } = &self.config.backend {
            let admin_config = ClientConfig::new()
                .set("bootstrap.servers", brokers.join(","))
                .clone();
            
            let admin_client: AdminClient<DefaultClientContext> = admin_config.create()
                .map_err(|e| anyhow!("Failed to create admin client: {}", e))?;
            
            let new_topic = NewTopic::new(&self.config.topic, 6, TopicReplication::Fixed(3))
                .set("cleanup.policy", "compact")
                .set("compression.type", "snappy")
                .set("min.insync.replicas", "2");
            
            let create_result = admin_client
                .create_topics(vec![&new_topic], &AdminOptions::new())
                .await;
            
            match create_result {
                Ok(_) => info!("Topic '{}' created successfully", self.config.topic),
                Err(e) => {
                    if e.to_string().contains("already exists") {
                        debug!("Topic '{}' already exists", self.config.topic);
                    } else {
                        return Err(anyhow!("Failed to create topic: {}", e));
                    }
                }
            }
        }
        Ok(())
    }
    
    pub async fn publish(&mut self, event: StreamEvent) -> Result<()> {
        let kafka_event = KafkaEvent::from(event);
        
        #[cfg(feature = "kafka")]
        {
            if let Some(ref producer) = self.producer {
                let payload = serde_json::to_string(&kafka_event)
                    .map_err(|e| anyhow!("Failed to serialize event: {}", e))?;
                
                let record = FutureRecord::to(&self.config.topic)
                    .key(&kafka_event.correlation_id)
                    .payload(&payload)
                    .timestamp(kafka_event.timestamp.timestamp_millis());
                
                match producer.send(record, Duration::from_secs(10)).await {
                    Ok(_) => {
                        self.stats.events_published += 1;
                        self.stats.bytes_sent += payload.len() as u64;
                        debug!("Published event: {}", kafka_event.correlation_id);
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
            self.pending_events.push(kafka_event);
            debug!("Mock publish: stored event in memory");
        }
        
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
                event_id: uuid::Uuid::new_v4().to_string(),
                timestamp: chrono::Utc::now(),
                source: "kafka_patch".to_string(),
                user: None,
                context: Some(patch.id.clone()),
                caused_by: None,
                version: "1.0".to_string(),
                properties: std::collections::HashMap::new(),
                checksum: None,
            };

            let event = match operation {
                PatchOperation::Add { subject, predicate, object } => {
                    StreamEvent::TripleAdded {
                        subject: subject.clone(),
                        predicate: predicate.clone(),
                        object: object.clone(),
                        graph: None,
                        metadata,
                    }
                }
                PatchOperation::Delete { subject, predicate, object } => {
                    StreamEvent::TripleRemoved {
                        subject: subject.clone(),
                        predicate: predicate.clone(),
                        object: object.clone(),
                        graph: None,
                        metadata,
                    }
                }
                PatchOperation::AddGraph { graph } => {
                    StreamEvent::GraphCreated {
                        graph: graph.clone(),
                        metadata,
                    }
                }
                PatchOperation::DeleteGraph { graph } => {
                    StreamEvent::GraphDeleted {
                        graph: graph.clone(),
                        metadata,
                    }
                }
            };
            self.publish(event).await?;
        }
        self.flush().await
    }
    
    pub async fn flush(&mut self) -> Result<()> {
        #[cfg(feature = "kafka")]
        {
            if let Some(ref producer) = self.producer {
                producer.flush(Duration::from_secs(10))
                    .map_err(|e| anyhow!("Failed to flush Kafka producer: {}", e))?;
                self.stats.last_flush = Some(chrono::Utc::now());
                debug!("Flushed Kafka producer");
            }
        }
        #[cfg(not(feature = "kafka"))]
        {
            debug!("Mock flush: {} pending events", self.pending_events.len());
        }
        Ok(())
    }
    
    pub fn get_stats(&self) -> &ProducerStats {
        &self.stats
    }
}

/// Kafka consumer for RDF streaming
pub struct KafkaConsumer {
    config: StreamConfig,
    #[cfg(feature = "kafka")]
    consumer: Option<KafkaStreamConsumer>,
    #[cfg(not(feature = "kafka"))]
    _phantom: std::marker::PhantomData<()>,
    stats: ConsumerStats,
}

#[derive(Debug, Default)]
struct ConsumerStats {
    events_consumed: u64,
    events_failed: u64,
    bytes_received: u64,
    last_message: Option<chrono::DateTime<chrono::Utc>>,
}

impl KafkaConsumer {
    pub fn new(config: StreamConfig) -> Result<Self> {
        #[cfg(feature = "kafka")]
        {
            let consumer = if let StreamBackend::Kafka { brokers } = &config.backend {
                let mut client_config = ClientConfig::new();
                client_config
                    .set("bootstrap.servers", brokers.join(","))
                    .set("group.id", "oxirs-consumer")
                    .set("enable.partition.eof", "false")
                    .set("session.timeout.ms", "30000")
                    .set("heartbeat.interval.ms", "10000")
                    .set("auto.offset.reset", "earliest")
                    .set("enable.auto.commit", "true")
                    .set("auto.commit.interval.ms", "5000")
                    .set("compression.type", "snappy");
                
                let consumer: KafkaStreamConsumer = client_config.create()
                    .map_err(|e| anyhow!("Failed to create Kafka consumer: {}", e))?;
                
                Some(consumer)
            } else {
                None
            };
            
            Ok(Self {
                config,
                consumer,
                stats: ConsumerStats::default(),
            })
        }
        #[cfg(not(feature = "kafka"))]
        {
            warn!("Kafka feature not enabled, using mock consumer");
            Ok(Self {
                config,
                _phantom: std::marker::PhantomData,
                stats: ConsumerStats::default(),
            })
        }
    }
    
    #[cfg(feature = "kafka")]
    pub async fn subscribe(&mut self) -> Result<()> {
        if let Some(ref consumer) = self.consumer {
            let mut topic_partition_list = TopicPartitionList::new();
            topic_partition_list.add_topic_unassigned(&self.config.topic);
            
            consumer.subscribe(&[&self.config.topic])
                .map_err(|e| anyhow!("Failed to subscribe to topic: {}", e))?;
            
            info!("Subscribed to Kafka topic: {}", self.config.topic);
        }
        Ok(())
    }
    
    pub async fn consume(&mut self) -> Result<Option<StreamEvent>> {
        #[cfg(feature = "kafka")]
        {
            if let Some(ref consumer) = self.consumer {
                match consumer.recv().await {
                    Ok(message) => {
                        if let Some(payload) = message.payload_view::<str>() {
                            match payload {
                                Ok(payload_str) => {
                                    match serde_json::from_str::<KafkaEvent>(payload_str) {
                                        Ok(kafka_event) => {
                                            self.stats.events_consumed += 1;
                                            self.stats.bytes_received += payload_str.len() as u64;
                                            self.stats.last_message = Some(chrono::Utc::now());
                                            
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
                        error!("Kafka consumer error: {}", e);
                        Err(anyhow!("Kafka receive error: {}", e))
                    }
                }
            } else {
                Err(anyhow!("Kafka consumer not initialized"))
            }
        }
        #[cfg(not(feature = "kafka"))]
        {
            // Mock consumer - return None to simulate no messages
            time::sleep(Duration::from_millis(100)).await;
            Ok(None)
        }
    }
    
    pub async fn consume_batch(&mut self, max_events: usize, timeout: Duration) -> Result<Vec<StreamEvent>> {
        let mut events = Vec::new();
        let start_time = time::Instant::now();
        
        while events.len() < max_events && start_time.elapsed() < timeout {
            match time::timeout(Duration::from_millis(100), self.consume()).await {
                Ok(Ok(Some(event))) => events.push(event),
                Ok(Ok(None)) => continue,
                Ok(Err(e)) => return Err(e),
                Err(_) => break, // Timeout
            }
        }
        
        Ok(events)
    }
    
    pub fn get_stats(&self) -> &ConsumerStats {
        &self.stats
    }
}

/// Kafka admin utilities
pub struct KafkaAdmin {
    #[cfg(feature = "kafka")]
    admin_client: AdminClient<DefaultClientContext>,
    #[cfg(not(feature = "kafka"))]
    _phantom: std::marker::PhantomData<()>,
}

impl KafkaAdmin {
    pub fn new(brokers: &[String]) -> Result<Self> {
        #[cfg(feature = "kafka")]
        {
            let admin_config = ClientConfig::new()
                .set("bootstrap.servers", brokers.join(","))
                .clone();
            
            let admin_client = admin_config.create()
                .map_err(|e| anyhow!("Failed to create Kafka admin client: {}", e))?;
            
            Ok(Self { admin_client })
        }
        #[cfg(not(feature = "kafka"))]
        {
            Ok(Self { _phantom: std::marker::PhantomData })
        }
    }
    
    #[cfg(feature = "kafka")]
    pub async fn list_topics(&self) -> Result<Vec<String>> {
        let metadata = self.admin_client.inner().fetch_metadata(None, Duration::from_secs(10))
            .map_err(|e| anyhow!("Failed to fetch metadata: {}", e))?;
        
        let topics = metadata.topics()
            .iter()
            .map(|topic| topic.name().to_string())
            .collect();
        
        Ok(topics)
    }
    
    #[cfg(not(feature = "kafka"))]
    pub async fn list_topics(&self) -> Result<Vec<String>> {
        Ok(vec!["mock-topic".to_string()])
    }
    
    #[cfg(feature = "kafka")]
    pub async fn create_topic(&self, topic: &str, partitions: i32, replication: i16) -> Result<()> {
        let new_topic = NewTopic::new(topic, partitions, TopicReplication::Fixed(replication as i32));
        
        let result = self.admin_client
            .create_topics(vec![&new_topic], &AdminOptions::new())
            .await;
        
        match result {
            Ok(_) => {
                info!("Successfully created topic: {}", topic);
                Ok(())
            }
            Err(e) => {
                if e.to_string().contains("already exists") {
                    debug!("Topic '{}' already exists", topic);
                    Ok(())
                } else {
                    Err(anyhow!("Failed to create topic '{}': {}", topic, e))
                }
            }
        }
    }
    
    #[cfg(not(feature = "kafka"))]
    pub async fn create_topic(&self, topic: &str, _partitions: i32, _replication: i16) -> Result<()> {
        info!("Mock: created topic {}", topic);
        Ok(())
    }
}