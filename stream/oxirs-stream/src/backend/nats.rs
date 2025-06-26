//! # NATS JetStream Backend
//!
//! NATS JetStream backend implementation for the streaming module.

use async_trait::async_trait;
use async_nats::{
    jetstream::{
        consumer::{pull::Config as PullConfig, Consumer},
        stream::{Config as StreamConfig, RetentionPolicy, StorageType, DiscardPolicy},
        Context, Message,
    },
    Client, ConnectOptions, Event, ServerAddr,
};
use bytes::Bytes;
use futures::StreamExt;
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

const DEFAULT_NATS_URL: &str = "nats://localhost:4222";
const DEFAULT_STREAM_PREFIX: &str = "OXIRS_";
const DEFAULT_SUBJECT_PREFIX: &str = "oxirs.";
const DEFAULT_MAX_MESSAGES: i64 = -1; // Unlimited
const DEFAULT_MAX_BYTES: i64 = -1; // Unlimited
const DEFAULT_MAX_AGE: Duration = Duration::from_secs(86400); // 24 hours
const DEFAULT_REPLICAS: usize = 1;
const DEFAULT_ACK_WAIT: Duration = Duration::from_secs(30);
const DEFAULT_MAX_DELIVER: i64 = 5;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NatsBackendConfig {
    pub urls: Vec<String>,
    pub user: Option<String>,
    pub password: Option<String>,
    pub token: Option<String>,
    pub nkey: Option<String>,
    pub jwt: Option<String>,
    pub credentials_path: Option<String>,
    pub stream_prefix: String,
    pub subject_prefix: String,
    pub max_reconnects: Option<usize>,
    pub reconnect_delay_ms: u64,
    pub connection_timeout_ms: u64,
    pub request_timeout_ms: u64,
    pub ping_interval_ms: u64,
    pub flush_interval_ms: u64,
    pub inbox_prefix: Option<String>,
    pub no_echo: bool,
    pub headers: bool,
    pub tls_required: bool,
    pub tls_cert_path: Option<String>,
    pub tls_key_path: Option<String>,
    pub tls_ca_path: Option<String>,
}

impl Default for NatsBackendConfig {
    fn default() -> Self {
        Self {
            urls: vec![DEFAULT_NATS_URL.to_string()],
            user: None,
            password: None,
            token: None,
            nkey: None,
            jwt: None,
            credentials_path: None,
            stream_prefix: DEFAULT_STREAM_PREFIX.to_string(),
            subject_prefix: DEFAULT_SUBJECT_PREFIX.to_string(),
            max_reconnects: Some(10),
            reconnect_delay_ms: 2000,
            connection_timeout_ms: 5000,
            request_timeout_ms: 5000,
            ping_interval_ms: 20000,
            flush_interval_ms: 100,
            inbox_prefix: None,
            no_echo: false,
            headers: true,
            tls_required: false,
            tls_cert_path: None,
            tls_key_path: None,
            tls_ca_path: None,
        }
    }
}

pub struct NatsBackend {
    config: NatsBackendConfig,
    client: Arc<Client>,
    jetstream: Arc<Context>,
    streams: Arc<RwLock<HashMap<String, async_nats::jetstream::stream::Stream>>>,
    consumers: Arc<Mutex<HashMap<String, Consumer<PullConfig>>>>,
    stats: Arc<RwLock<NatsStats>>,
}

#[derive(Clone, Debug, Default)]
struct NatsStats {
    messages_sent: u64,
    messages_received: u64,
    messages_failed: u64,
    bytes_sent: u64,
    bytes_received: u64,
    ack_failures: u64,
    reconnections: u64,
    last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NatsEventMessage {
    event_id: String,
    event_type: String,
    timestamp: u64,
    data: serde_json::Value,
    metadata: Option<EventMetadata>,
}

impl From<StreamEvent> for NatsEventMessage {
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

        Self {
            event_id: Uuid::new_v4().to_string(),
            event_type,
            timestamp: event.timestamp,
            data,
            metadata: event.metadata,
        }
    }
}

impl NatsEventMessage {
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

impl NatsBackend {
    pub async fn new(config: NatsBackendConfig) -> StreamResult<Self> {
        let mut connect_options = ConnectOptions::new();

        // Set connection options
        if let Some(user) = &config.user {
            connect_options = connect_options.user(user.clone());
        }
        if let Some(password) = &config.password {
            connect_options = connect_options.password(password.clone());
        }
        if let Some(token) = &config.token {
            connect_options = connect_options.token(token.clone());
        }
        if let Some(creds_path) = &config.credentials_path {
            connect_options = connect_options.credentials_path(creds_path).await
                .map_err(|e| StreamError::Configuration(format!("Failed to load credentials: {}", e)))?;
        }

        connect_options = connect_options
            .connection_timeout(Duration::from_millis(config.connection_timeout_ms))
            .ping_interval(Duration::from_millis(config.ping_interval_ms))
            .request_timeout(Some(Duration::from_millis(config.request_timeout_ms)))
            .require_tls(config.tls_required);

        if let Some(max_reconnects) = config.max_reconnects {
            connect_options = connect_options.max_reconnects(max_reconnects);
        }

        // Convert URLs to ServerAddr
        let servers: Vec<ServerAddr> = config.urls
            .iter()
            .map(|url| url.parse::<ServerAddr>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| StreamError::Configuration(format!("Invalid NATS URL: {}", e)))?;

        // Connect to NATS
        let client = connect_options
            .connect(servers)
            .await
            .map_err(|e| StreamError::Connection(format!("Failed to connect to NATS: {}", e)))?;

        // Create JetStream context
        let jetstream = async_nats::jetstream::new(client.clone());

        Ok(Self {
            config,
            client: Arc::new(client),
            jetstream: Arc::new(jetstream),
            streams: Arc::new(RwLock::new(HashMap::new())),
            consumers: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(RwLock::new(NatsStats::default())),
        })
    }

    fn get_stream_name(&self, topic: &TopicName) -> String {
        format!("{}{}", self.config.stream_prefix, topic.as_str().to_uppercase().replace('-', "_"))
    }

    fn get_subject_name(&self, topic: &TopicName) -> String {
        format!("{}{}", self.config.subject_prefix, topic.as_str())
    }

    async fn ensure_stream(&self, stream_name: &str, subject: &str) -> StreamResult<async_nats::jetstream::stream::Stream> {
        let streams = self.streams.read().await;
        if let Some(stream) = streams.get(stream_name) {
            return Ok(stream.clone());
        }
        drop(streams);

        // Create stream configuration
        let stream_config = StreamConfig::builder()
            .name(stream_name)
            .subjects(vec![subject.to_string()])
            .retention(RetentionPolicy::Limits)
            .storage(StorageType::File)
            .max_messages(DEFAULT_MAX_MESSAGES)
            .max_bytes(DEFAULT_MAX_BYTES)
            .max_age(DEFAULT_MAX_AGE)
            .num_replicas(DEFAULT_REPLICAS)
            .discard(DiscardPolicy::Old)
            .build()
            .map_err(|e| StreamError::Configuration(format!("Failed to build stream config: {}", e)))?;

        // Create or update stream
        let stream = self.jetstream
            .get_or_create_stream(stream_config)
            .await
            .map_err(|e| StreamError::Backend(format!("Failed to create stream: {}", e)))?;

        self.streams.write().await.insert(stream_name.to_string(), stream.clone());
        info!("Created/verified NATS JetStream: {}", stream_name);

        Ok(stream)
    }

    async fn get_or_create_consumer(
        &self,
        stream_name: &str,
        consumer_group: Option<&ConsumerGroup>,
        position: StreamPosition,
    ) -> StreamResult<Consumer<PullConfig>> {
        let consumer_name = consumer_group
            .map(|g| g.name().to_string())
            .unwrap_or_else(|| format!("oxirs-consumer-{}", Uuid::new_v4()));

        let consumer_key = format!("{}:{}", stream_name, consumer_name);
        
        let mut consumers = self.consumers.lock().await;
        if let Some(consumer) = consumers.get(&consumer_key) {
            return Ok(consumer.clone());
        }

        // Get the stream
        let stream = self.jetstream
            .get_stream(stream_name)
            .await
            .map_err(|e| StreamError::Backend(format!("Failed to get stream: {}", e)))?;

        // Create consumer configuration
        let mut consumer_config = PullConfig::builder()
            .durable_name(consumer_name.clone())
            .description("OxiRS Stream Consumer".to_string())
            .ack_wait(DEFAULT_ACK_WAIT)
            .max_deliver(DEFAULT_MAX_DELIVER);

        // Set deliver policy based on position
        consumer_config = match position {
            StreamPosition::Beginning => consumer_config.deliver_all(),
            StreamPosition::End => consumer_config.deliver_new(),
            StreamPosition::Offset(offset) => consumer_config.deliver_by_start_sequence(offset + 1),
        };

        let consumer_config = consumer_config
            .build()
            .map_err(|e| StreamError::Configuration(format!("Failed to build consumer config: {}", e)))?;

        // Create consumer
        let consumer = stream
            .get_or_create_consumer(&consumer_name, consumer_config)
            .await
            .map_err(|e| StreamError::Backend(format!("Failed to create consumer: {}", e)))?;

        consumers.insert(consumer_key, consumer.clone());
        Ok(consumer)
    }
}

#[async_trait]
impl StreamBackend for NatsBackend {
    fn name(&self) -> &'static str {
        "nats"
    }

    async fn connect(&mut self) -> StreamResult<()> {
        // Already connected in new()
        info!("NATS backend connected to: {:?}", self.config.urls);
        Ok(())
    }

    async fn disconnect(&mut self) -> StreamResult<()> {
        self.streams.write().await.clear();
        self.consumers.lock().await.clear();
        info!("Disconnected from NATS");
        Ok(())
    }

    async fn create_topic(&self, topic: &TopicName, _partitions: u32) -> StreamResult<()> {
        let stream_name = self.get_stream_name(topic);
        let subject = self.get_subject_name(topic);
        
        self.ensure_stream(&stream_name, &subject).await?;
        Ok(())
    }

    async fn delete_topic(&self, topic: &TopicName) -> StreamResult<()> {
        let stream_name = self.get_stream_name(topic);
        
        match self.jetstream.delete_stream(&stream_name).await {
            Ok(_) => {
                self.streams.write().await.remove(&stream_name);
                info!("Deleted NATS stream: {}", stream_name);
                Ok(())
            }
            Err(e) => Err(StreamError::Backend(format!("Failed to delete stream: {}", e))),
        }
    }

    async fn list_topics(&self) -> StreamResult<Vec<TopicName>> {
        let mut topics = Vec::new();
        let stream_names = self.jetstream.stream_names();
        
        let mut stream_names = Box::pin(stream_names);
        while let Some(result) = stream_names.next().await {
            match result {
                Ok(stream_name) => {
                    if let Some(topic_name) = stream_name.strip_prefix(&self.config.stream_prefix) {
                        let topic_name = topic_name.to_lowercase().replace('_', "-");
                        topics.push(TopicName::new(topic_name));
                    }
                }
                Err(e) => {
                    warn!("Error listing streams: {}", e);
                }
            }
        }
        
        Ok(topics)
    }

    async fn send_event(&self, topic: &TopicName, event: StreamEvent) -> StreamResult<Offset> {
        let subject = self.get_subject_name(topic);
        let stream_name = self.get_stream_name(topic);
        
        // Ensure stream exists
        self.ensure_stream(&stream_name, &subject).await?;
        
        // Convert event to NATS message
        let nats_event = NatsEventMessage::from(event);
        let payload = serde_json::to_vec(&nats_event)
            .map_err(|e| StreamError::Serialization(e.to_string()))?;
        
        // Publish message
        let ack = self.jetstream
            .publish(subject, Bytes::from(payload.clone()))
            .await
            .map_err(|e| StreamError::Backend(format!("Failed to publish message: {}", e)))?
            .await
            .map_err(|e| StreamError::Backend(format!("Failed to get publish ack: {}", e)))?;
        
        self.stats.write().await.messages_sent += 1;
        self.stats.write().await.bytes_sent += payload.len() as u64;
        
        Ok(Offset::new(ack.sequence))
    }

    async fn send_batch(&self, topic: &TopicName, events: Vec<StreamEvent>) -> StreamResult<Vec<Offset>> {
        let subject = self.get_subject_name(topic);
        let stream_name = self.get_stream_name(topic);
        
        // Ensure stream exists
        self.ensure_stream(&stream_name, &subject).await?;
        
        let mut offsets = Vec::with_capacity(events.len());
        
        for event in events {
            let nats_event = NatsEventMessage::from(event);
            let payload = serde_json::to_vec(&nats_event)
                .map_err(|e| StreamError::Serialization(e.to_string()))?;
            
            match self.jetstream
                .publish(subject.clone(), Bytes::from(payload.clone()))
                .await {
                Ok(future) => {
                    match future.await {
                        Ok(ack) => {
                            offsets.push(Offset::new(ack.sequence));
                            self.stats.write().await.messages_sent += 1;
                            self.stats.write().await.bytes_sent += payload.len() as u64;
                        }
                        Err(e) => {
                            self.stats.write().await.messages_failed += 1;
                            error!("Failed to get publish ack: {}", e);
                        }
                    }
                }
                Err(e) => {
                    self.stats.write().await.messages_failed += 1;
                    error!("Failed to publish message: {}", e);
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
        let stream_name = self.get_stream_name(topic);
        let consumer = self.get_or_create_consumer(&stream_name, consumer_group, position).await?;
        
        let mut events = Vec::new();
        
        // Fetch messages
        let messages = consumer
            .messages()
            .await
            .map_err(|e| StreamError::Backend(format!("Failed to get message stream: {}", e)))?;
        
        let mut messages = Box::pin(messages.take(max_events));
        
        while let Some(message_result) = messages.next().await {
            match message_result {
                Ok(message) => {
                    match serde_json::from_slice::<NatsEventMessage>(&message.payload) {
                        Ok(nats_event) => {
                            match nats_event.to_stream_event() {
                                Ok(stream_event) => {
                                    let offset = Offset::new(message.info().unwrap().stream_sequence);
                                    events.push((stream_event, offset));
                                    
                                    // Acknowledge message
                                    if let Err(e) = message.ack().await {
                                        warn!("Failed to acknowledge message: {}", e);
                                        self.stats.write().await.ack_failures += 1;
                                    }
                                    
                                    self.stats.write().await.messages_received += 1;
                                    self.stats.write().await.bytes_received += message.payload.len() as u64;
                                }
                                Err(e) => {
                                    error!("Failed to convert NATS event to stream event: {}", e);
                                    self.stats.write().await.messages_failed += 1;
                                }
                            }
                        }
                        Err(e) => {
                            error!("Failed to deserialize NATS message: {}", e);
                            self.stats.write().await.messages_failed += 1;
                        }
                    }
                }
                Err(e) => {
                    error!("Error receiving message: {}", e);
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
        // In NATS JetStream, acknowledgment happens per message during receive
        // This is handled in receive_events method
        debug!("NATS uses per-message acknowledgment. Offset {} already committed for group {} on topic {}", 
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
        let stream_name = self.get_stream_name(topic);
        let consumer_name = consumer_group.name();
        let consumer_key = format!("{}:{}", stream_name, consumer_name);
        
        // Remove existing consumer to force recreation with new position
        self.consumers.lock().await.remove(&consumer_key);
        
        info!("Reset consumer position for group {} on stream {} to {:?}", 
            consumer_name, stream_name, position);
        Ok(())
    }

    async fn get_consumer_lag(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
    ) -> StreamResult<HashMap<PartitionId, u64>> {
        let stream_name = self.get_stream_name(topic);
        let consumer_name = consumer_group.name();
        
        let mut lag_map = HashMap::new();
        
        // Get stream info
        let stream = self.jetstream
            .get_stream(&stream_name)
            .await
            .map_err(|e| StreamError::Backend(format!("Failed to get stream: {}", e)))?;
        
        let stream_info = stream.info().await
            .map_err(|e| StreamError::Backend(format!("Failed to get stream info: {}", e)))?;
        
        // Get consumer info
        match stream.consumer_info(consumer_name).await {
            Ok(consumer_info) => {
                let lag = stream_info.state.messages - consumer_info.delivered.stream_sequence;
                lag_map.insert(PartitionId::new(0), lag);
            }
            Err(_) => {
                // Consumer doesn't exist yet, lag is total messages
                lag_map.insert(PartitionId::new(0), stream_info.state.messages);
            }
        }
        
        Ok(lag_map)
    }

    async fn get_topic_metadata(&self, topic: &TopicName) -> StreamResult<HashMap<String, String>> {
        let stream_name = self.get_stream_name(topic);
        let mut metadata = HashMap::new();
        
        metadata.insert("backend".to_string(), "nats".to_string());
        metadata.insert("stream_name".to_string(), stream_name.clone());
        
        // Get stream info if it exists
        match self.jetstream.get_stream(&stream_name).await {
            Ok(stream) => {
                match stream.info().await {
                    Ok(info) => {
                        metadata.insert("messages".to_string(), info.state.messages.to_string());
                        metadata.insert("bytes".to_string(), info.state.bytes.to_string());
                        metadata.insert("consumer_count".to_string(), info.state.consumer_count.to_string());
                        metadata.insert("first_sequence".to_string(), info.state.first_sequence.to_string());
                        metadata.insert("last_sequence".to_string(), info.state.last_sequence.to_string());
                        
                        if let Some(created) = info.created {
                            metadata.insert("created".to_string(), created.to_string());
                        }
                    }
                    Err(e) => {
                        warn!("Failed to get stream info: {}", e);
                    }
                }
            }
            Err(e) => {
                warn!("Stream {} not found: {}", stream_name, e);
            }
        }
        
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
    fn test_nats_event_conversion() {
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

        let nats_event = NatsEventMessage::from(event.clone());
        assert_eq!(nats_event.event_type, "triple_added");
        assert_eq!(nats_event.timestamp, 12345);

        let converted_event = nats_event.to_stream_event().unwrap();
        assert_eq!(converted_event.timestamp, event.timestamp);
    }

    #[tokio::test]
    async fn test_nats_backend_creation() {
        let config = NatsBackendConfig::default();
        let backend = NatsBackend::new(config).await;
        // May fail if NATS is not running locally
        if backend.is_err() {
            println!("NATS backend creation failed - likely no local NATS instance");
        }
    }
}