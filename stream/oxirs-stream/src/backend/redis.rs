//! # Redis Streams Backend
//!
//! Redis Streams backend implementation for the streaming module.

use async_trait::async_trait;
use redis::aio::{ConnectionManager, MultiplexedConnection};
use redis::streams::{
    StreamId, StreamKey, StreamMaxlen, StreamPendingCountReply, StreamPendingReply, StreamReadOptions,
    StreamReadReply,
};
use redis::{AsyncCommands, Client, Cmd, RedisError, RedisResult, Value};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock};
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

use crate::backend::StreamBackend;
use crate::consumer::{ConsumerConfig, ConsumerGroup};
use crate::error::{StreamError, StreamResult};
use crate::event::{StreamEvent, StreamEventType};
use crate::producer::ProducerConfig;
use crate::types::{EventMetadata, Offset, PartitionId, StreamPosition, TopicName};

const DEFAULT_REDIS_URL: &str = "redis://127.0.0.1:6379/";
const DEFAULT_MAX_RETRIES: u32 = 3;
const DEFAULT_RETRY_DELAY_MS: u64 = 100;
const DEFAULT_STREAM_MAXLEN: usize = 1_000_000;
const DEFAULT_BLOCK_MS: usize = 1000;
const DEFAULT_COUNT: usize = 100;
const CONSUMER_GROUP_PREFIX: &str = "oxirs:cg:";
const DEAD_LETTER_SUFFIX: &str = ":dlq";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RedisBackendConfig {
    pub url: String,
    pub connection_pool_size: u32,
    pub max_retries: u32,
    pub retry_delay_ms: u64,
    pub stream_maxlen: usize,
    pub block_ms: usize,
    pub count: usize,
    pub enable_cluster: bool,
    pub username: Option<String>,
    pub password: Option<String>,
    pub db: i64,
    pub read_timeout_ms: u64,
    pub write_timeout_ms: u64,
    pub connection_timeout_ms: u64,
}

impl Default for RedisBackendConfig {
    fn default() -> Self {
        Self {
            url: DEFAULT_REDIS_URL.to_string(),
            connection_pool_size: 10,
            max_retries: DEFAULT_MAX_RETRIES,
            retry_delay_ms: DEFAULT_RETRY_DELAY_MS,
            stream_maxlen: DEFAULT_STREAM_MAXLEN,
            block_ms: DEFAULT_BLOCK_MS,
            count: DEFAULT_COUNT,
            enable_cluster: false,
            username: None,
            password: None,
            db: 0,
            read_timeout_ms: 5000,
            write_timeout_ms: 5000,
            connection_timeout_ms: 5000,
        }
    }
}

pub struct RedisBackend {
    config: RedisBackendConfig,
    client: Arc<Client>,
    connections: Arc<RwLock<Vec<ConnectionManager>>>,
    consumer_groups: Arc<Mutex<HashMap<String, ConsumerGroupState>>>,
    stats: Arc<RwLock<RedisStats>>,
}

#[derive(Clone, Debug, Default)]
struct ConsumerGroupState {
    group_name: String,
    consumer_id: String,
    pending_count: usize,
    last_id: String,
    active: bool,
}

#[derive(Clone, Debug, Default)]
struct RedisStats {
    messages_sent: u64,
    messages_received: u64,
    messages_failed: u64,
    bytes_sent: u64,
    bytes_received: u64,
    connection_errors: u64,
    last_error: Option<String>,
}

impl RedisBackend {
    pub async fn new(config: RedisBackendConfig) -> StreamResult<Self> {
        let client = Client::open(config.url.as_str())
            .map_err(|e| StreamError::Backend(format!("Failed to create Redis client: {}", e)))?;

        let mut connections = Vec::with_capacity(config.connection_pool_size as usize);
        
        for _ in 0..config.connection_pool_size {
            let conn = ConnectionManager::new(client.clone())
                .await
                .map_err(|e| StreamError::Backend(format!("Failed to create connection: {}", e)))?;
            connections.push(conn);
        }

        Ok(Self {
            config,
            client: Arc::new(client),
            connections: Arc::new(RwLock::new(connections)),
            consumer_groups: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(RwLock::new(RedisStats::default())),
        })
    }

    async fn get_connection(&self) -> StreamResult<ConnectionManager> {
        let connections = self.connections.read().await;
        if let Some(conn) = connections.first() {
            Ok(conn.clone())
        } else {
            Err(StreamError::Backend("No connections available".to_string()))
        }
    }

    async fn execute_with_retry<F, T>(&self, mut operation: F) -> StreamResult<T>
    where
        F: FnMut() -> RedisResult<T>,
    {
        let mut retries = 0;
        loop {
            match operation() {
                Ok(result) => return Ok(result),
                Err(e) => {
                    retries += 1;
                    if retries > self.config.max_retries {
                        self.stats.write().await.connection_errors += 1;
                        self.stats.write().await.last_error = Some(e.to_string());
                        return Err(StreamError::Backend(format!("Redis error after {} retries: {}", retries, e)));
                    }
                    warn!("Redis operation failed, retrying ({}/{}): {}", retries, self.config.max_retries, e);
                    sleep(Duration::from_millis(self.config.retry_delay_ms * retries as u64)).await;
                }
            }
        }
    }

    fn serialize_event(&self, event: &StreamEvent) -> StreamResult<HashMap<String, String>> {
        let mut fields = HashMap::new();
        
        fields.insert("event_type".to_string(), format!("{:?}", event.event_type));
        fields.insert("timestamp".to_string(), event.timestamp.to_string());
        
        if let Some(metadata) = &event.metadata {
            fields.insert("source".to_string(), metadata.source.clone());
            if let Some(user) = &metadata.user {
                fields.insert("user".to_string(), user.clone());
            }
            if let Some(session) = &metadata.session_id {
                fields.insert("session_id".to_string(), session.clone());
            }
            if let Some(trace) = &metadata.trace_id {
                fields.insert("trace_id".to_string(), trace.clone());
            }
        }

        let event_data = serde_json::to_string(&event.event_type)
            .map_err(|e| StreamError::Serialization(e.to_string()))?;
        fields.insert("data".to_string(), event_data);

        Ok(fields)
    }

    fn deserialize_event(&self, id: &str, fields: &HashMap<String, Value>) -> StreamResult<StreamEvent> {
        let data = fields.get("data")
            .and_then(|v| v.as_string())
            .ok_or_else(|| StreamError::Deserialization("Missing data field".to_string()))?;

        let event_type: StreamEventType = serde_json::from_str(data)
            .map_err(|e| StreamError::Deserialization(e.to_string()))?;

        let timestamp = fields.get("timestamp")
            .and_then(|v| v.as_string())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64);

        let metadata = if fields.contains_key("source") {
            Some(EventMetadata {
                source: fields.get("source")
                    .and_then(|v| v.as_string())
                    .unwrap_or_default()
                    .to_string(),
                user: fields.get("user")
                    .and_then(|v| v.as_string())
                    .map(|s| s.to_string()),
                session_id: fields.get("session_id")
                    .and_then(|v| v.as_string())
                    .map(|s| s.to_string()),
                trace_id: fields.get("trace_id")
                    .and_then(|v| v.as_string())
                    .map(|s| s.to_string()),
                causality_token: None,
                version: None,
            })
        } else {
            None
        };

        Ok(StreamEvent {
            event_type,
            timestamp,
            metadata,
        })
    }

    async fn ensure_consumer_group(&self, stream_key: &str, group_name: &str) -> StreamResult<()> {
        let mut conn = self.get_connection().await?;
        
        let result: RedisResult<String> = conn.xgroup_create_mkstream(stream_key, group_name, "$").await;
        
        match result {
            Ok(_) => {
                info!("Created consumer group {} for stream {}", group_name, stream_key);
                Ok(())
            }
            Err(e) => {
                if e.to_string().contains("BUSYGROUP") {
                    debug!("Consumer group {} already exists for stream {}", group_name, stream_key);
                    Ok(())
                } else {
                    Err(StreamError::Backend(format!("Failed to create consumer group: {}", e)))
                }
            }
        }
    }

    async fn handle_dead_letter(&self, stream_key: &str, message_id: &str, fields: &HashMap<String, Value>) -> StreamResult<()> {
        let dlq_key = format!("{}{}", stream_key, DEAD_LETTER_SUFFIX);
        let mut conn = self.get_connection().await?;

        let mut dlq_fields: Vec<(String, String)> = fields.iter()
            .map(|(k, v)| (k.clone(), v.to_string()))
            .collect();
        
        dlq_fields.push(("original_stream".to_string(), stream_key.to_string()));
        dlq_fields.push(("original_id".to_string(), message_id.to_string()));
        dlq_fields.push(("dlq_timestamp".to_string(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis().to_string()));

        let _: String = conn.xadd(&dlq_key, "*", &dlq_fields).await
            .map_err(|e| StreamError::Backend(format!("Failed to add to dead letter queue: {}", e)))?;

        warn!("Message {} from stream {} moved to dead letter queue", message_id, stream_key);
        Ok(())
    }
}

#[async_trait]
impl StreamBackend for RedisBackend {
    fn name(&self) -> &'static str {
        "redis"
    }

    async fn connect(&mut self) -> StreamResult<()> {
        let mut conn = self.get_connection().await?;
        let _: String = conn.ping().await
            .map_err(|e| StreamError::Connection(format!("Failed to ping Redis: {}", e)))?;
        
        info!("Successfully connected to Redis backend");
        Ok(())
    }

    async fn disconnect(&mut self) -> StreamResult<()> {
        self.connections.write().await.clear();
        self.consumer_groups.lock().await.clear();
        info!("Disconnected from Redis backend");
        Ok(())
    }

    async fn create_topic(&self, topic: &TopicName, _partitions: u32) -> StreamResult<()> {
        let mut conn = self.get_connection().await?;
        
        let key = format!("oxirs:stream:{}", topic);
        let exists: bool = conn.exists(&key).await
            .map_err(|e| StreamError::Backend(format!("Failed to check topic existence: {}", e)))?;

        if !exists {
            let _: String = conn.xadd(&key, "*", &[("init", "true")]).await
                .map_err(|e| StreamError::Backend(format!("Failed to create topic: {}", e)))?;
            
            let _: usize = conn.xdel(&key, &["0-0"]).await
                .map_err(|e| StreamError::Backend(format!("Failed to clean init message: {}", e)))?;
                
            info!("Created Redis stream topic: {}", topic);
        }

        Ok(())
    }

    async fn delete_topic(&self, topic: &TopicName) -> StreamResult<()> {
        let mut conn = self.get_connection().await?;
        let key = format!("oxirs:stream:{}", topic);
        
        let _: bool = conn.del(&key).await
            .map_err(|e| StreamError::Backend(format!("Failed to delete topic: {}", e)))?;
            
        info!("Deleted Redis stream topic: {}", topic);
        Ok(())
    }

    async fn list_topics(&self) -> StreamResult<Vec<TopicName>> {
        let mut conn = self.get_connection().await?;
        
        let keys: Vec<String> = conn.keys("oxirs:stream:*").await
            .map_err(|e| StreamError::Backend(format!("Failed to list topics: {}", e)))?;

        let topics: Vec<TopicName> = keys.iter()
            .filter_map(|key| {
                key.strip_prefix("oxirs:stream:")
                    .map(|name| TopicName::new(name.to_string()))
            })
            .collect();

        Ok(topics)
    }

    async fn send_event(&self, topic: &TopicName, event: StreamEvent) -> StreamResult<Offset> {
        let mut conn = self.get_connection().await?;
        let key = format!("oxirs:stream:{}", topic);
        
        let fields = self.serialize_event(&event)?;
        let field_pairs: Vec<(String, String)> = fields.into_iter().collect();
        
        let id: String = conn.xadd(&key, "*", &field_pairs).await
            .map_err(|e| StreamError::Backend(format!("Failed to send event: {}", e)))?;

        let event_size = serde_json::to_vec(&event).unwrap_or_default().len();
        self.stats.write().await.messages_sent += 1;
        self.stats.write().await.bytes_sent += event_size as u64;

        Ok(Offset::new(id.parse::<u64>().unwrap_or(0)))
    }

    async fn send_batch(&self, topic: &TopicName, events: Vec<StreamEvent>) -> StreamResult<Vec<Offset>> {
        let mut offsets = Vec::with_capacity(events.len());
        
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
        let mut conn = self.get_connection().await?;
        let key = format!("oxirs:stream:{}", topic);
        
        let mut events = Vec::new();

        if let Some(group) = consumer_group {
            let group_name = format!("{}{}", CONSUMER_GROUP_PREFIX, group.name());
            let consumer_id = group.consumer_id().unwrap_or("default");
            
            self.ensure_consumer_group(&key, &group_name).await?;

            let start_id = match position {
                StreamPosition::Beginning => "0".to_string(),
                StreamPosition::End => ">".to_string(),
                StreamPosition::Offset(offset) => offset.to_string(),
            };

            let read_reply: StreamReadReply = conn.xreadgroup_options(
                &group_name,
                consumer_id,
                &[&key],
                &[&start_id],
                &StreamReadOptions::default()
                    .count(max_events)
                    .block(self.config.block_ms),
            ).await
            .map_err(|e| StreamError::Backend(format!("Failed to read from consumer group: {}", e)))?;

            for stream_key in read_reply.keys {
                for msg in stream_key.ids {
                    match self.deserialize_event(&msg.id, &msg.map) {
                        Ok(event) => {
                            let offset = Offset::new(msg.id.parse::<u64>().unwrap_or(0));
                            events.push((event, offset));
                            self.stats.write().await.messages_received += 1;
                        }
                        Err(e) => {
                            error!("Failed to deserialize message {}: {}", msg.id, e);
                            self.handle_dead_letter(&key, &msg.id, &msg.map).await?;
                            self.stats.write().await.messages_failed += 1;
                        }
                    }
                }
            }
        } else {
            let start_id = match position {
                StreamPosition::Beginning => "-".to_string(),
                StreamPosition::End => "+".to_string(),
                StreamPosition::Offset(offset) => offset.to_string(),
            };

            let read_reply: StreamReadReply = conn.xread_options(
                &[&key],
                &[&start_id],
                &StreamReadOptions::default()
                    .count(max_events)
                    .block(self.config.block_ms),
            ).await
            .map_err(|e| StreamError::Backend(format!("Failed to read from stream: {}", e)))?;

            for stream_key in read_reply.keys {
                for msg in stream_key.ids {
                    match self.deserialize_event(&msg.id, &msg.map) {
                        Ok(event) => {
                            let offset = Offset::new(msg.id.parse::<u64>().unwrap_or(0));
                            events.push((event, offset));
                            self.stats.write().await.messages_received += 1;
                        }
                        Err(e) => {
                            error!("Failed to deserialize message {}: {}", msg.id, e);
                            self.stats.write().await.messages_failed += 1;
                        }
                    }
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
        let mut conn = self.get_connection().await?;
        let key = format!("oxirs:stream:{}", topic);
        let group_name = format!("{}{}", CONSUMER_GROUP_PREFIX, consumer_group.name());
        
        let message_id = offset.to_string();
        let ack_count: usize = conn.xack(&key, &group_name, &[&message_id]).await
            .map_err(|e| StreamError::Backend(format!("Failed to acknowledge message: {}", e)))?;

        if ack_count == 0 {
            warn!("No messages acknowledged for offset {} in topic {}", offset, topic);
        } else {
            debug!("Acknowledged message {} in topic {} for group {}", message_id, topic, group_name);
        }

        Ok(())
    }

    async fn seek(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
        _partition: PartitionId,
        position: StreamPosition,
    ) -> StreamResult<()> {
        let group_name = format!("{}{}", CONSUMER_GROUP_PREFIX, consumer_group.name());
        let mut groups = self.consumer_groups.lock().await;
        
        let state = groups.entry(group_name.clone()).or_insert_with(|| ConsumerGroupState {
            group_name: group_name.clone(),
            consumer_id: consumer_group.consumer_id().unwrap_or("default").to_string(),
            pending_count: 0,
            last_id: match position {
                StreamPosition::Beginning => "0".to_string(),
                StreamPosition::End => "$".to_string(),
                StreamPosition::Offset(offset) => offset.to_string(),
            },
            active: true,
        });

        state.last_id = match position {
            StreamPosition::Beginning => "0".to_string(),
            StreamPosition::End => "$".to_string(),
            StreamPosition::Offset(offset) => offset.to_string(),
        };

        info!("Seek consumer group {} to position {:?} for topic {}", group_name, position, topic);
        Ok(())
    }

    async fn get_consumer_lag(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
    ) -> StreamResult<HashMap<PartitionId, u64>> {
        let mut conn = self.get_connection().await?;
        let key = format!("oxirs:stream:{}", topic);
        let group_name = format!("{}{}", CONSUMER_GROUP_PREFIX, consumer_group.name());
        
        let mut lag_map = HashMap::new();

        let info: StreamPendingReply = conn.xpending(&key, &group_name).await
            .map_err(|e| StreamError::Backend(format!("Failed to get consumer lag: {}", e)))?;

        lag_map.insert(PartitionId::new(0), info.count() as u64);

        Ok(lag_map)
    }

    async fn get_topic_metadata(&self, topic: &TopicName) -> StreamResult<HashMap<String, String>> {
        let mut conn = self.get_connection().await?;
        let key = format!("oxirs:stream:{}", topic);
        
        let mut metadata = HashMap::new();

        let len: usize = conn.xlen(&key).await
            .map_err(|e| StreamError::Backend(format!("Failed to get stream length: {}", e)))?;
        
        metadata.insert("length".to_string(), len.to_string());
        metadata.insert("backend".to_string(), "redis".to_string());
        metadata.insert("stream_key".to_string(), key);

        let info: Value = conn.xinfo_stream(&key).await
            .map_err(|e| StreamError::Backend(format!("Failed to get stream info: {}", e)))?;

        if let Value::Bulk(items) = info {
            for chunk in items.chunks(2) {
                if let (Some(Value::Data(key)), Some(value)) = (chunk.get(0), chunk.get(1)) {
                    if let Ok(key_str) = String::from_utf8(key.clone()) {
                        metadata.insert(key_str, value.to_string());
                    }
                }
            }
        }

        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::StreamEventType;

    #[tokio::test]
    async fn test_redis_backend_creation() {
        let config = RedisBackendConfig::default();
        let backend = RedisBackend::new(config).await;
        assert!(backend.is_ok());
    }

    #[tokio::test]
    async fn test_event_serialization() {
        let config = RedisBackendConfig::default();
        let backend = RedisBackend::new(config).await.unwrap();
        
        let event = StreamEvent {
            event_type: StreamEventType::TripleAdded {
                subject: "http://example.org/subject".to_string(),
                predicate: "http://example.org/predicate".to_string(),
                object: "http://example.org/object".to_string(),
                graph: None,
            },
            timestamp: 12345,
            metadata: Some(EventMetadata {
                source: "test".to_string(),
                user: Some("user1".to_string()),
                session_id: Some("session1".to_string()),
                trace_id: Some("trace1".to_string()),
                causality_token: None,
                version: None,
            }),
        };

        let fields = backend.serialize_event(&event).unwrap();
        assert!(fields.contains_key("event_type"));
        assert!(fields.contains_key("timestamp"));
        assert!(fields.contains_key("data"));
        assert!(fields.contains_key("source"));
        assert!(fields.contains_key("user"));
    }
}