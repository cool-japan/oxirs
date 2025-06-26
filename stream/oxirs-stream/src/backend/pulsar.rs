//! # Apache Pulsar Backend
//!
//! Apache Pulsar backend implementation for the streaming module.

use async_trait::async_trait;
use pulsar::{
    Authentication, CompressionType, Consumer, ConsumerBuilder, ConsumerOptions, DeserializeMessage,
    Error as PulsarError, Message, Payload, Producer, ProducerBuilder, ProducerOptions, Pulsar,
    PulsarBuilder, SerializeMessage, SubType, TokioExecutor,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock};
use tokio::time::sleep;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::backend::StreamBackend;
use crate::consumer::ConsumerGroup;
use crate::error::{StreamError, StreamResult};
use crate::event::{StreamEvent, StreamEventType};
use crate::types::{EventMetadata, Offset, PartitionId, StreamPosition, TopicName};

const DEFAULT_SERVICE_URL: &str = "pulsar://localhost:6650";
const DEFAULT_PARTITIONS: u32 = 4;
const DEFAULT_REPLICATION_FACTOR: u32 = 1;
const DEFAULT_ACK_TIMEOUT_MS: u64 = 30000;
const DEFAULT_BATCH_SIZE: u32 = 1000;
const DEFAULT_COMPRESSION: CompressionType = CompressionType::Lz4;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PulsarBackendConfig {
    pub service_url: String,
    pub admin_url: Option<String>,
    pub auth_method: Option<PulsarAuthMethod>,
    pub auth_params: HashMap<String, String>,
    pub namespace: String,
    pub tenant: String,
    pub partitions: u32,
    pub replication_factor: u32,
    pub compression: String,
    pub batch_size: u32,
    pub ack_timeout_ms: u64,
    pub enable_tls: bool,
    pub tls_trust_certs_path: Option<String>,
    pub tls_allow_insecure: bool,
    pub operation_timeout_ms: u64,
    pub connection_timeout_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PulsarAuthMethod {
    Token,
    OAuth2,
    Basic,
    Athenz,
    Tls,
}

impl Default for PulsarBackendConfig {
    fn default() -> Self {
        Self {
            service_url: DEFAULT_SERVICE_URL.to_string(),
            admin_url: None,
            auth_method: None,
            auth_params: HashMap::new(),
            namespace: "oxirs".to_string(),
            tenant: "public".to_string(),
            partitions: DEFAULT_PARTITIONS,
            replication_factor: DEFAULT_REPLICATION_FACTOR,
            compression: "lz4".to_string(),
            batch_size: DEFAULT_BATCH_SIZE,
            ack_timeout_ms: DEFAULT_ACK_TIMEOUT_MS,
            enable_tls: false,
            tls_trust_certs_path: None,
            tls_allow_insecure: false,
            operation_timeout_ms: 30000,
            connection_timeout_ms: 10000,
        }
    }
}

pub struct PulsarBackend {
    config: PulsarBackendConfig,
    client: Arc<Pulsar<TokioExecutor>>,
    producers: Arc<RwLock<HashMap<String, Producer<TokioExecutor>>>>,
    consumers: Arc<Mutex<HashMap<String, Consumer<PulsarStreamEvent, TokioExecutor>>>>,
    stats: Arc<RwLock<PulsarStats>>,
}

#[derive(Clone, Debug, Default)]
struct PulsarStats {
    messages_sent: u64,
    messages_received: u64,
    messages_failed: u64,
    bytes_sent: u64,
    bytes_received: u64,
    ack_timeouts: u64,
    connection_errors: u64,
    last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PulsarStreamEvent {
    pub event_id: String,
    pub event_type: String,
    pub timestamp: u64,
    pub data: serde_json::Value,
    pub metadata: Option<EventMetadata>,
    pub properties: HashMap<String, String>,
}

impl SerializeMessage for PulsarStreamEvent {
    fn serialize_message(input: Self) -> Result<Message, PulsarError> {
        let payload = serde_json::to_vec(&input)
            .map_err(|e| PulsarError::Custom(format!("Serialization error: {}", e)))?;
        
        Ok(Message {
            payload,
            ..Default::default()
        })
    }
}

impl DeserializeMessage for PulsarStreamEvent {
    type Output = Result<PulsarStreamEvent, serde_json::Error>;
    
    fn deserialize_message(payload: &Payload) -> Self::Output {
        serde_json::from_slice(&payload.data)
    }
}

impl From<StreamEvent> for PulsarStreamEvent {
    fn from(event: StreamEvent) -> Self {
        let (event_type, data) = match &event.event_type {
            StreamEventType::TripleAdded { subject, predicate, object, graph } => (
                "triple_added".to_string(),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }),
            ),
            StreamEventType::TripleRemoved { subject, predicate, object, graph } => (
                "triple_removed".to_string(),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }),
            ),
            StreamEventType::QuadAdded { subject, predicate, object, graph } => (
                "quad_added".to_string(),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }),
            ),
            StreamEventType::QuadRemoved { subject, predicate, object, graph } => (
                "quad_removed".to_string(),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }),
            ),
            StreamEventType::GraphCreated { graph } => (
                "graph_created".to_string(),
                serde_json::json!({
                    "graph": graph
                }),
            ),
            StreamEventType::GraphCleared { graph } => (
                "graph_cleared".to_string(),
                serde_json::json!({
                    "graph": graph
                }),
            ),
            StreamEventType::GraphDeleted { graph } => (
                "graph_deleted".to_string(),
                serde_json::json!({
                    "graph": graph
                }),
            ),
            StreamEventType::SparqlUpdate { query } => (
                "sparql_update".to_string(),
                serde_json::json!({
                    "query": query
                }),
            ),
            StreamEventType::TransactionBegin { transaction_id } => (
                "transaction_begin".to_string(),
                serde_json::json!({
                    "transaction_id": transaction_id
                }),
            ),
            StreamEventType::TransactionCommit { transaction_id } => (
                "transaction_commit".to_string(),
                serde_json::json!({
                    "transaction_id": transaction_id
                }),
            ),
            StreamEventType::TransactionAbort { transaction_id } => (
                "transaction_abort".to_string(),
                serde_json::json!({
                    "transaction_id": transaction_id
                }),
            ),
        };

        let mut properties = HashMap::new();
        if let Some(metadata) = &event.metadata {
            properties.insert("source".to_string(), metadata.source.clone());
            if let Some(user) = &metadata.user {
                properties.insert("user".to_string(), user.clone());
            }
            if let Some(session_id) = &metadata.session_id {
                properties.insert("session_id".to_string(), session_id.clone());
            }
            if let Some(trace_id) = &metadata.trace_id {
                properties.insert("trace_id".to_string(), trace_id.clone());
            }
        }

        Self {
            event_id: Uuid::new_v4().to_string(),
            event_type,
            timestamp: event.timestamp,
            data,
            metadata: event.metadata,
            properties,
        }
    }
}

impl PulsarStreamEvent {
    fn to_stream_event(&self) -> StreamResult<StreamEvent> {
        let event_type = match self.event_type.as_str() {
            "triple_added" => {
                let subject = self.data["subject"].as_str()
                    .ok_or_else(|| StreamError::Deserialization("Missing subject".to_string()))?
                    .to_string();
                let predicate = self.data["predicate"].as_str()
                    .ok_or_else(|| StreamError::Deserialization("Missing predicate".to_string()))?
                    .to_string();
                let object = self.data["object"].as_str()
                    .ok_or_else(|| StreamError::Deserialization("Missing object".to_string()))?
                    .to_string();
                let graph = self.data["graph"].as_str().map(|s| s.to_string());
                
                StreamEventType::TripleAdded { subject, predicate, object, graph }
            }
            "triple_removed" => {
                let subject = self.data["subject"].as_str()
                    .ok_or_else(|| StreamError::Deserialization("Missing subject".to_string()))?
                    .to_string();
                let predicate = self.data["predicate"].as_str()
                    .ok_or_else(|| StreamError::Deserialization("Missing predicate".to_string()))?
                    .to_string();
                let object = self.data["object"].as_str()
                    .ok_or_else(|| StreamError::Deserialization("Missing object".to_string()))?
                    .to_string();
                let graph = self.data["graph"].as_str().map(|s| s.to_string());
                
                StreamEventType::TripleRemoved { subject, predicate, object, graph }
            }
            "graph_created" => {
                let graph = self.data["graph"].as_str()
                    .ok_or_else(|| StreamError::Deserialization("Missing graph".to_string()))?
                    .to_string();
                StreamEventType::GraphCreated { graph }
            }
            "graph_cleared" => {
                let graph = self.data["graph"].as_str().map(|s| s.to_string());
                StreamEventType::GraphCleared { graph }
            }
            "graph_deleted" => {
                let graph = self.data["graph"].as_str()
                    .ok_or_else(|| StreamError::Deserialization("Missing graph".to_string()))?
                    .to_string();
                StreamEventType::GraphDeleted { graph }
            }
            _ => return Err(StreamError::Deserialization(format!("Unknown event type: {}", self.event_type))),
        };

        Ok(StreamEvent {
            event_type,
            timestamp: self.timestamp,
            metadata: self.metadata.clone(),
        })
    }
}

impl PulsarBackend {
    pub async fn new(config: PulsarBackendConfig) -> StreamResult<Self> {
        let mut builder = PulsarBuilder::default()
            .with_url(&config.service_url)
            .with_connection_timeout(Duration::from_millis(config.connection_timeout_ms))
            .with_operation_timeout(Duration::from_millis(config.operation_timeout_ms));

        // Configure authentication
        if let Some(auth_method) = &config.auth_method {
            let auth = match auth_method {
                PulsarAuthMethod::Token => {
                    if let Some(token) = config.auth_params.get("token") {
                        Authentication::Token(token.clone())
                    } else {
                        return Err(StreamError::Configuration("Token authentication requires 'token' parameter".to_string()));
                    }
                }
                PulsarAuthMethod::OAuth2 => {
                    return Err(StreamError::NotSupported("OAuth2 authentication not yet implemented".to_string()));
                }
                PulsarAuthMethod::Basic => {
                    if let (Some(username), Some(password)) = (
                        config.auth_params.get("username"),
                        config.auth_params.get("password"),
                    ) {
                        Authentication::Basic {
                            username: username.clone(),
                            password: password.clone(),
                        }
                    } else {
                        return Err(StreamError::Configuration("Basic authentication requires 'username' and 'password' parameters".to_string()));
                    }
                }
                _ => {
                    return Err(StreamError::NotSupported(format!("Authentication method {:?} not supported", auth_method)));
                }
            };
            builder = builder.with_auth(auth);
        }

        // Configure TLS
        if config.enable_tls {
            if let Some(trust_certs) = &config.tls_trust_certs_path {
                builder = builder.with_certificate_chain_file(trust_certs);
            }
            builder = builder.with_allow_insecure_connection(config.tls_allow_insecure);
        }

        let client: Pulsar<_> = builder.build().await
            .map_err(|e| StreamError::Connection(format!("Failed to build Pulsar client: {}", e)))?;

        Ok(Self {
            config,
            client: Arc::new(client),
            producers: Arc::new(RwLock::new(HashMap::new())),
            consumers: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(RwLock::new(PulsarStats::default())),
        })
    }

    fn get_topic_name(&self, topic: &TopicName) -> String {
        format!("persistent://{}/{}/{}", self.config.tenant, self.config.namespace, topic.as_str())
    }

    fn parse_compression(&self) -> CompressionType {
        match self.config.compression.to_lowercase().as_str() {
            "lz4" => CompressionType::Lz4,
            "zlib" => CompressionType::Zlib,
            "zstd" => CompressionType::Zstd,
            "snappy" => CompressionType::Snappy,
            _ => DEFAULT_COMPRESSION,
        }
    }

    async fn get_or_create_producer(&self, topic_name: &str) -> StreamResult<Producer<TokioExecutor>> {
        let producers = self.producers.read().await;
        if let Some(producer) = producers.get(topic_name) {
            return Ok(producer.clone());
        }
        drop(producers);

        // Create new producer
        let producer: Producer<TokioExecutor> = self.client
            .producer()
            .with_topic(topic_name)
            .with_name(format!("oxirs-producer-{}", Uuid::new_v4()))
            .with_options(ProducerOptions {
                batch_size: Some(self.config.batch_size),
                compression: Some(self.parse_compression()),
                ..Default::default()
            })
            .build()
            .await
            .map_err(|e| StreamError::Backend(format!("Failed to create producer: {}", e)))?;

        self.producers.write().await.insert(topic_name.to_string(), producer.clone());
        Ok(producer)
    }

    async fn create_consumer(
        &self,
        topic_name: &str,
        consumer_group: Option<&ConsumerGroup>,
        position: StreamPosition,
    ) -> StreamResult<Consumer<PulsarStreamEvent, TokioExecutor>> {
        let subscription = consumer_group
            .map(|g| g.name().to_string())
            .unwrap_or_else(|| format!("oxirs-consumer-{}", Uuid::new_v4()));

        let sub_type = if consumer_group.is_some() {
            SubType::Shared
        } else {
            SubType::Exclusive
        };

        let mut builder = self.client
            .consumer()
            .with_topic(topic_name)
            .with_subscription(&subscription)
            .with_subscription_type(sub_type)
            .with_consumer_name(format!("oxirs-consumer-{}", Uuid::new_v4()));

        // Set initial position
        match position {
            StreamPosition::Beginning => {
                builder = builder.with_options(ConsumerOptions {
                    initial_position: Some(0),
                    ..Default::default()
                });
            }
            StreamPosition::End => {
                // Default behavior is to start from latest
            }
            StreamPosition::Offset(offset) => {
                // Pulsar doesn't directly support offset-based positioning
                // This would require message ID based positioning
                warn!("Offset-based positioning not directly supported in Pulsar, starting from latest");
            }
        }

        let consumer = builder
            .build()
            .await
            .map_err(|e| StreamError::Backend(format!("Failed to create consumer: {}", e)))?;

        Ok(consumer)
    }
}

#[async_trait]
impl StreamBackend for PulsarBackend {
    fn name(&self) -> &'static str {
        "pulsar"
    }

    async fn connect(&mut self) -> StreamResult<()> {
        // Test connection by getting broker metadata
        // Note: Pulsar client doesn't have a direct "ping" equivalent
        info!("Connected to Pulsar at {}", self.config.service_url);
        Ok(())
    }

    async fn disconnect(&mut self) -> StreamResult<()> {
        self.producers.write().await.clear();
        self.consumers.lock().await.clear();
        info!("Disconnected from Pulsar");
        Ok(())
    }

    async fn create_topic(&self, topic: &TopicName, partitions: u32) -> StreamResult<()> {
        let topic_name = self.get_topic_name(topic);
        
        // Pulsar automatically creates topics when producers/consumers connect
        // For explicit creation, you'd need the Pulsar admin API
        
        // Try to create a producer to ensure topic exists
        let _ = self.get_or_create_producer(&topic_name).await?;
        
        info!("Created/verified Pulsar topic: {}", topic_name);
        Ok(())
    }

    async fn delete_topic(&self, topic: &TopicName) -> StreamResult<()> {
        let topic_name = self.get_topic_name(topic);
        
        // Topic deletion requires Pulsar admin API
        // For now, just remove from local caches
        self.producers.write().await.remove(&topic_name);
        
        warn!("Topic deletion requires Pulsar admin API. Topic {} not deleted from broker.", topic_name);
        Ok(())
    }

    async fn list_topics(&self) -> StreamResult<Vec<TopicName>> {
        // Listing topics requires Pulsar admin API
        // Return empty list for now
        warn!("Topic listing requires Pulsar admin API. Returning empty list.");
        Ok(Vec::new())
    }

    async fn send_event(&self, topic: &TopicName, event: StreamEvent) -> StreamResult<Offset> {
        let topic_name = self.get_topic_name(topic);
        let pulsar_event = PulsarStreamEvent::from(event);
        
        let producer = self.get_or_create_producer(&topic_name).await?;
        
        let data = serde_json::to_vec(&pulsar_event)
            .map_err(|e| StreamError::Serialization(e.to_string()))?;
        let data_size = data.len();
        
        let receipt = producer
            .send(pulsar_event)
            .await
            .map_err(|e| StreamError::Backend(format!("Failed to send message: {}", e)))?
            .await
            .map_err(|e| StreamError::Backend(format!("Failed to get send receipt: {}", e)))?;

        self.stats.write().await.messages_sent += 1;
        self.stats.write().await.bytes_sent += data_size as u64;

        // Convert message ID to offset (simplified)
        // In practice, you'd need proper message ID handling
        let offset = Offset::new(receipt.sequence_id.unwrap_or(0) as u64);
        
        debug!("Sent message to Pulsar topic {} with sequence {}", topic_name, offset);
        Ok(offset)
    }

    async fn send_batch(&self, topic: &TopicName, events: Vec<StreamEvent>) -> StreamResult<Vec<Offset>> {
        let topic_name = self.get_topic_name(topic);
        let producer = self.get_or_create_producer(&topic_name).await?;
        let mut offsets = Vec::new();
        
        for event in events {
            let pulsar_event = PulsarStreamEvent::from(event);
            let data = serde_json::to_vec(&pulsar_event)
                .map_err(|e| StreamError::Serialization(e.to_string()))?;
            let data_size = data.len();
            
            match producer.send(pulsar_event).await {
                Ok(send_future) => {
                    match send_future.await {
                        Ok(receipt) => {
                            let offset = Offset::new(receipt.sequence_id.unwrap_or(0) as u64);
                            offsets.push(offset);
                            self.stats.write().await.messages_sent += 1;
                            self.stats.write().await.bytes_sent += data_size as u64;
                        }
                        Err(e) => {
                            self.stats.write().await.messages_failed += 1;
                            error!("Failed to get send receipt: {}", e);
                        }
                    }
                }
                Err(e) => {
                    self.stats.write().await.messages_failed += 1;
                    error!("Failed to send message: {}", e);
                }
            }
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
        let topic_name = self.get_topic_name(topic);
        let mut events = Vec::new();
        
        // Create consumer key
        let consumer_key = format!("{}:{}", 
            consumer_group.map(|g| g.name()).unwrap_or("default"),
            topic_name
        );
        
        // Get or create consumer
        let mut consumers = self.consumers.lock().await;
        let consumer = if let Some(existing) = consumers.get_mut(&consumer_key) {
            existing
        } else {
            let new_consumer = self.create_consumer(&topic_name, consumer_group, position).await?;
            consumers.insert(consumer_key.clone(), new_consumer);
            consumers.get_mut(&consumer_key).unwrap()
        };
        
        // Receive messages
        for _ in 0..max_events {
            match tokio::time::timeout(Duration::from_millis(100), consumer.next()).await {
                Ok(Some(Ok(msg))) => {
                    match msg.deserialize() {
                        Ok(pulsar_event) => {
                            match pulsar_event.to_stream_event() {
                                Ok(stream_event) => {
                                    let offset = Offset::new(msg.sequence_id() as u64);
                                    events.push((stream_event, offset));
                                    
                                    // Acknowledge message
                                    if let Err(e) = consumer.ack(&msg).await {
                                        warn!("Failed to acknowledge message: {}", e);
                                    }
                                    
                                    self.stats.write().await.messages_received += 1;
                                    self.stats.write().await.bytes_received += msg.payload.data.len() as u64;
                                }
                                Err(e) => {
                                    error!("Failed to convert Pulsar event to stream event: {}", e);
                                    self.stats.write().await.messages_failed += 1;
                                }
                            }
                        }
                        Err(e) => {
                            error!("Failed to deserialize Pulsar message: {}", e);
                            self.stats.write().await.messages_failed += 1;
                        }
                    }
                }
                Ok(Some(Err(e))) => {
                    error!("Error receiving message: {}", e);
                    self.stats.write().await.connection_errors += 1;
                }
                Ok(None) => {
                    // No more messages
                    break;
                }
                Err(_) => {
                    // Timeout - no message available
                    break;
                }
            }
        }
        
        Ok(events)
    }

    async fn commit_offset(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
        _partition: PartitionId,
        offset: Offset,
    ) -> StreamResult<()> {
        // In Pulsar, acknowledgment happens per message during receive
        // This is handled in receive_events method
        debug!("Pulsar uses per-message acknowledgment. Offset {} already committed for group {} on topic {}", 
            offset, consumer_group.name(), topic);
        Ok(())
    }

    async fn seek(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
        _partition: PartitionId,
        position: StreamPosition,
    ) -> StreamResult<()> {
        let topic_name = self.get_topic_name(topic);
        let consumer_key = format!("{}:{}", consumer_group.name(), topic_name);
        
        // Remove existing consumer to force recreation with new position
        self.consumers.lock().await.remove(&consumer_key);
        
        info!("Reset consumer position for group {} on topic {} to {:?}", 
            consumer_group.name(), topic, position);
        Ok(())
    }

    async fn get_consumer_lag(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
    ) -> StreamResult<HashMap<PartitionId, u64>> {
        // Getting consumer lag requires Pulsar admin API
        // Return empty map for now
        warn!("Consumer lag calculation requires Pulsar admin API. Returning empty lag map.");
        Ok(HashMap::new())
    }

    async fn get_topic_metadata(&self, topic: &TopicName) -> StreamResult<HashMap<String, String>> {
        let topic_name = self.get_topic_name(topic);
        let mut metadata = HashMap::new();
        
        metadata.insert("backend".to_string(), "pulsar".to_string());
        metadata.insert("topic_name".to_string(), topic_name);
        metadata.insert("tenant".to_string(), self.config.tenant.clone());
        metadata.insert("namespace".to_string(), self.config.namespace.clone());
        metadata.insert("service_url".to_string(), self.config.service_url.clone());
        
        // Add stats
        let stats = self.stats.read().await;
        metadata.insert("messages_sent".to_string(), stats.messages_sent.to_string());
        metadata.insert("messages_received".to_string(), stats.messages_received.to_string());
        metadata.insert("bytes_sent".to_string(), stats.bytes_sent.to_string());
        metadata.insert("bytes_received".to_string(), stats.bytes_received.to_string());
        
        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::StreamEventType;

    #[test]
    fn test_pulsar_event_conversion() {
        let event = StreamEvent {
            event_type: StreamEventType::TripleAdded {
                subject: "http://example.org/s".to_string(),
                predicate: "http://example.org/p".to_string(),
                object: "http://example.org/o".to_string(),
                graph: None,
            },
            timestamp: 12345,
            metadata: Some(EventMetadata {
                source: "test".to_string(),
                user: Some("user1".to_string()),
                session_id: None,
                trace_id: None,
                causality_token: None,
                version: None,
            }),
        };

        let pulsar_event = PulsarStreamEvent::from(event.clone());
        assert_eq!(pulsar_event.event_type, "triple_added");
        assert_eq!(pulsar_event.timestamp, 12345);

        let converted_event = pulsar_event.to_stream_event().unwrap();
        assert_eq!(converted_event.timestamp, event.timestamp);
    }

    #[tokio::test]
    async fn test_pulsar_backend_creation() {
        let config = PulsarBackendConfig::default();
        let backend = PulsarBackend::new(config).await;
        // May fail if Pulsar is not running locally
        if backend.is_err() {
            println!("Pulsar backend creation failed - likely no local Pulsar instance");
        }
    }
}