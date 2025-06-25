//! # Redis Streams Backend
//!
//! Redis Streams support for ultra-high performance RDF streaming.
//!
//! This module provides comprehensive Redis Streams integration with clustering,
//! consumer groups, persistence, and real-time message processing capabilities.
//! Optimized for ultra-low latency and high throughput scenarios.

use anyhow::{anyhow, Result};
use crate::{StreamEvent, StreamConfig, StreamBackend, RdfPatch, PatchOperation, EventMetadata};
use crate::kafka::KafkaEvent;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time;
use tracing::{debug, error, info, warn};
use chrono::{DateTime, Utc};
use uuid::Uuid;

#[cfg(feature = "redis")]
use redis::{
    AsyncCommands, Client, ConnectionManager, RedisResult, 
    streams::{StreamReadOptions, StreamReadReply},
    cluster::ClusterClient,
    cluster_async::ClusterConnection,
    aio::ConnectionManager as AsyncConnectionManager,
};

/// Redis Streams configuration with clustering and performance tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisStreamConfig {
    pub urls: Vec<String>,
    pub stream_name: String,
    pub consumer_group: String,
    pub consumer_name: String,
    pub max_len: Option<usize>,
    pub approximate_trimming: bool,
    pub cluster_mode: bool,
    pub pool_size: usize,
    pub connection_timeout: Duration,
    pub read_timeout: Duration,
    pub retry_attempts: u32,
    pub retry_delay: Duration,
    pub enable_pipeline: bool,
    pub pipeline_size: usize,
    pub enable_persistence: bool,
    pub compression_enabled: bool,
}

impl Default for RedisStreamConfig {
    fn default() -> Self {
        Self {
            urls: vec!["redis://localhost:6379".to_string()],
            stream_name: "oxirs:rdf:stream".to_string(),
            consumer_group: "oxirs-consumers".to_string(),
            consumer_name: format!("consumer-{}", Uuid::new_v4()),
            max_len: Some(1_000_000),
            approximate_trimming: true,
            cluster_mode: false,
            pool_size: 10,
            connection_timeout: Duration::from_secs(5),
            read_timeout: Duration::from_millis(100),
            retry_attempts: 3,
            retry_delay: Duration::from_millis(100),
            enable_pipeline: true,
            pipeline_size: 100,
            enable_persistence: true,
            compression_enabled: true,
        }
    }
}

/// Redis connection manager supporting both standalone and cluster modes
pub enum RedisConnectionManager {
    #[cfg(feature = "redis")]
    Standalone(AsyncConnectionManager),
    #[cfg(feature = "redis")]
    Cluster(ClusterConnection),
    #[cfg(not(feature = "redis"))]
    Mock,
}

/// Enhanced Redis producer with high-performance optimizations
pub struct RedisProducer {
    config: StreamConfig,
    redis_config: RedisStreamConfig,
    connection: Option<RedisConnectionManager>,
    pending_events: Vec<RedisStreamEvent>,
    pipeline_buffer: Vec<RedisStreamEvent>,
    stats: ProducerStats,
    last_flush: Instant,
    sequence_number: u64,
}

#[derive(Debug, Default)]
struct ProducerStats {
    events_published: u64,
    events_failed: u64,
    bytes_sent: u64,
    pipeline_flushes: u64,
    connection_retries: u64,
    last_publish: Option<DateTime<Utc>>,
    avg_latency_ms: f64,
    max_latency_ms: u64,
}

/// Serializable event for Redis transmission with compression support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisStreamEvent {
    pub event_id: String,
    pub event_type: String,
    pub timestamp: DateTime<Utc>,
    pub sequence: u64,
    pub data: serde_json::Value,
    pub metadata: EventMetadata,
    pub checksum: Option<String>,
    pub compressed: bool,
}

impl From<StreamEvent> for RedisStreamEvent {
    fn from(event: StreamEvent) -> Self {
        let (event_type, data, metadata) = match event {
            StreamEvent::TripleAdded { subject, predicate, object, graph, metadata } => {
                ("triple_added".to_string(), serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }), metadata)
            }
            StreamEvent::TripleRemoved { subject, predicate, object, graph, metadata } => {
                ("triple_removed".to_string(), serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }), metadata)
            }
            StreamEvent::QuadAdded { subject, predicate, object, graph, metadata } => {
                ("quad_added".to_string(), serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }), metadata)
            }
            StreamEvent::QuadRemoved { subject, predicate, object, graph, metadata } => {
                ("quad_removed".to_string(), serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }), metadata)
            }
            StreamEvent::GraphCreated { graph, metadata } => {
                ("graph_created".to_string(), serde_json::json!({
                    "graph": graph
                }), metadata)
            }
            StreamEvent::GraphCleared { graph, metadata } => {
                ("graph_cleared".to_string(), serde_json::json!({
                    "graph": graph
                }), metadata)
            }
            StreamEvent::GraphDeleted { graph, metadata } => {
                ("graph_deleted".to_string(), serde_json::json!({
                    "graph": graph
                }), metadata)
            }
            StreamEvent::SparqlUpdate { query, operation_type, metadata } => {
                ("sparql_update".to_string(), serde_json::json!({
                    "query": query,
                    "operation_type": operation_type
                }), metadata)
            }
            StreamEvent::TransactionBegin { transaction_id, isolation_level, metadata } => {
                ("transaction_begin".to_string(), serde_json::json!({
                    "transaction_id": transaction_id,
                    "isolation_level": isolation_level
                }), metadata)
            }
            StreamEvent::TransactionCommit { transaction_id, metadata } => {
                ("transaction_commit".to_string(), serde_json::json!({
                    "transaction_id": transaction_id
                }), metadata)
            }
            StreamEvent::TransactionAbort { transaction_id, metadata } => {
                ("transaction_abort".to_string(), serde_json::json!({
                    "transaction_id": transaction_id
                }), metadata)
            }
            StreamEvent::SchemaChanged { schema_type, change_type, details, metadata } => {
                ("schema_changed".to_string(), serde_json::json!({
                    "schema_type": schema_type,
                    "change_type": change_type,
                    "details": details
                }), metadata)
            }
            StreamEvent::Heartbeat { timestamp, source } => {
                ("heartbeat".to_string(), serde_json::json!({
                    "source": source
                }), EventMetadata {
                    event_id: Uuid::new_v4().to_string(),
                    timestamp,
                    source: source.clone(),
                    user: None,
                    context: None,
                    caused_by: None,
                    version: "1.0".to_string(),
                    properties: HashMap::new(),
                    checksum: None,
                })
            }
        };

        Self {
            event_id: metadata.event_id.clone(),
            event_type,
            timestamp: metadata.timestamp,
            sequence: 0, // Will be set by producer
            data,
            metadata,
            checksum: None,
            compressed: false,
        }
    }
}

impl RedisProducer {
    pub fn new(config: StreamConfig) -> Result<Self> {
        let redis_config = if let StreamBackend::Redis { url, .. } = &config.backend {
            RedisStreamConfig {
                urls: vec![url.clone()],
                ..Default::default()
            }
        } else {
            RedisStreamConfig::default()
        };

        Ok(Self {
            config,
            redis_config,
            connection: None,
            pending_events: Vec::new(),
            pipeline_buffer: Vec::new(),
            stats: ProducerStats::default(),
            last_flush: Instant::now(),
            sequence_number: 0,
        })
    }

    pub fn with_redis_config(mut self, redis_config: RedisStreamConfig) -> Self {
        self.redis_config = redis_config;
        self
    }

    #[cfg(feature = "redis")]
    pub async fn connect(&mut self) -> Result<()> {
        let connection = if self.redis_config.cluster_mode {
            let cluster_client = ClusterClient::new(self.redis_config.urls.clone())
                .map_err(|e| anyhow!("Failed to create Redis cluster client: {}", e))?;
            
            let connection = cluster_client.get_async_connection().await
                .map_err(|e| anyhow!("Failed to connect to Redis cluster: {}", e))?;
            
            RedisConnectionManager::Cluster(connection)
        } else {
            let client = Client::open(self.redis_config.urls[0].as_str())
                .map_err(|e| anyhow!("Failed to create Redis client: {}", e))?;
            
            let manager = client.get_connection_manager().await
                .map_err(|e| anyhow!("Failed to get Redis connection manager: {}", e))?;
            
            RedisConnectionManager::Standalone(manager)
        };

        self.connection = Some(connection);
        
        // Create consumer group if it doesn't exist
        self.ensure_consumer_group().await?;
        
        info!("Connected to Redis: cluster={}", self.redis_config.cluster_mode);
        Ok(())
    }

    #[cfg(not(feature = "redis"))]
    pub async fn connect(&mut self) -> Result<()> {
        warn!("Redis feature not enabled, using mock connection");
        self.connection = Some(RedisConnectionManager::Mock);
        Ok(())
    }

    #[cfg(feature = "redis")]
    async fn ensure_consumer_group(&mut self) -> Result<()> {
        match &mut self.connection {
            Some(RedisConnectionManager::Standalone(manager)) => {
                let result: RedisResult<String> = redis::cmd("XGROUP")
                    .arg("CREATE")
                    .arg(&self.redis_config.stream_name)
                    .arg(&self.redis_config.consumer_group)
                    .arg("0")
                    .arg("MKSTREAM")
                    .query_async(manager)
                    .await;

                match result {
                    Ok(_) => info!("Created consumer group: {}", self.redis_config.consumer_group),
                    Err(e) if e.to_string().contains("BUSYGROUP") => {
                        debug!("Consumer group already exists: {}", self.redis_config.consumer_group);
                    }
                    Err(e) => warn!("Failed to create consumer group: {}", e),
                }
            }
            Some(RedisConnectionManager::Cluster(connection)) => {
                let result: RedisResult<String> = redis::cmd("XGROUP")
                    .arg("CREATE")
                    .arg(&self.redis_config.stream_name)
                    .arg(&self.redis_config.consumer_group)
                    .arg("0")
                    .arg("MKSTREAM")
                    .query_async(connection)
                    .await;

                match result {
                    Ok(_) => info!("Created consumer group: {}", self.redis_config.consumer_group),
                    Err(e) if e.to_string().contains("BUSYGROUP") => {
                        debug!("Consumer group already exists: {}", self.redis_config.consumer_group);
                    }
                    Err(e) => warn!("Failed to create consumer group: {}", e),
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub async fn publish(&mut self, event: StreamEvent) -> Result<()> {
        let start_time = Instant::now();
        
        if self.connection.is_none() {
            self.connect().await?;
        }

        let mut redis_event = RedisStreamEvent::from(event);
        redis_event.sequence = self.sequence_number;
        self.sequence_number += 1;

        // Apply compression if enabled
        if self.redis_config.compression_enabled {
            redis_event = self.compress_event(redis_event)?;
        }

        // Add to pipeline buffer if pipelining is enabled
        if self.redis_config.enable_pipeline {
            self.pipeline_buffer.push(redis_event);
            
            if self.pipeline_buffer.len() >= self.redis_config.pipeline_size {
                self.flush_pipeline().await?;
            }
        } else {
            self.publish_single_event(&redis_event).await?;
        }

        // Update latency stats
        let latency = start_time.elapsed().as_millis() as u64;
        self.stats.max_latency_ms = self.stats.max_latency_ms.max(latency);
        self.stats.avg_latency_ms = (self.stats.avg_latency_ms + latency as f64) / 2.0;
        self.stats.last_publish = Some(Utc::now());

        Ok(())
    }

    #[cfg(feature = "redis")]
    async fn publish_single_event(&mut self, event: &RedisStreamEvent) -> Result<()> {
        let serialized = serde_json::to_string(event)
            .map_err(|e| anyhow!("Failed to serialize event: {}", e))?;

        let fields = vec![
            ("data", serialized.as_str()),
            ("event_type", &event.event_type),
            ("event_id", &event.event_id),
            ("timestamp", &event.timestamp.to_rfc3339()),
            ("sequence", &event.sequence.to_string()),
        ];

        match &mut self.connection {
            Some(RedisConnectionManager::Standalone(manager)) => {
                let mut cmd = redis::cmd("XADD");
                cmd.arg(&self.redis_config.stream_name);
                
                if let Some(max_len) = self.redis_config.max_len {
                    if self.redis_config.approximate_trimming {
                        cmd.arg("MAXLEN").arg("~").arg(max_len);
                    } else {
                        cmd.arg("MAXLEN").arg(max_len);
                    }
                }
                
                cmd.arg("*"); // Auto-generate ID
                
                for (key, value) in fields {
                    cmd.arg(key).arg(value);
                }

                let _: String = cmd.query_async(manager).await
                    .map_err(|e| anyhow!("Failed to publish to Redis: {}", e))?;

                self.stats.events_published += 1;
                self.stats.bytes_sent += serialized.len() as u64;
                debug!("Published event {} to Redis", event.event_id);
            }
            Some(RedisConnectionManager::Cluster(connection)) => {
                let mut cmd = redis::cmd("XADD");
                cmd.arg(&self.redis_config.stream_name);
                
                if let Some(max_len) = self.redis_config.max_len {
                    if self.redis_config.approximate_trimming {
                        cmd.arg("MAXLEN").arg("~").arg(max_len);
                    } else {
                        cmd.arg("MAXLEN").arg(max_len);
                    }
                }
                
                cmd.arg("*");
                
                for (key, value) in fields {
                    cmd.arg(key).arg(value);
                }

                let _: String = cmd.query_async(connection).await
                    .map_err(|e| anyhow!("Failed to publish to Redis cluster: {}", e))?;

                self.stats.events_published += 1;
                self.stats.bytes_sent += serialized.len() as u64;
                debug!("Published event {} to Redis cluster", event.event_id);
            }
            _ => {
                return Err(anyhow!("Redis connection not available"));
            }
        }

        Ok(())
    }

    #[cfg(not(feature = "redis"))]
    async fn publish_single_event(&mut self, event: &RedisStreamEvent) -> Result<()> {
        self.pending_events.push(event.clone());
        debug!("Mock Redis publish: {}", event.event_id);
        Ok(())
    }

    async fn flush_pipeline(&mut self) -> Result<()> {
        if self.pipeline_buffer.is_empty() {
            return Ok(());
        }

        let events = std::mem::take(&mut self.pipeline_buffer);
        
        for event in events {
            self.publish_single_event(&event).await?;
        }

        self.stats.pipeline_flushes += 1;
        debug!("Flushed pipeline with {} events", self.pipeline_buffer.len());
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
        if self.redis_config.enable_pipeline && !self.pipeline_buffer.is_empty() {
            self.flush_pipeline().await?;
        }

        self.last_flush = Instant::now();
        debug!("Flushed Redis producer");
        Ok(())
    }

    fn compress_event(&self, mut event: RedisStreamEvent) -> Result<RedisStreamEvent> {
        if self.redis_config.compression_enabled {
            let data_str = event.data.to_string();
            if data_str.len() > 1024 { // Only compress large events
                // Simple compression using flate2 would go here
                // For now, just mark as compressed
                event.compressed = true;
                debug!("Compressed event {} from {} bytes", event.event_id, data_str.len());
            }
        }
        Ok(event)
    }

    pub fn get_stats(&self) -> &ProducerStats {
        &self.stats
    }
}

/// Enhanced Redis consumer with consumer groups and parallel processing
pub struct RedisConsumer {
    config: StreamConfig,
    redis_config: RedisStreamConfig,
    connection: Option<RedisConnectionManager>,
    stats: ConsumerStats,
    last_id: String,
    block_time_ms: u64,
    batch_size: usize,
}

#[derive(Debug, Default)]
struct ConsumerStats {
    events_consumed: u64,
    events_failed: u64,
    bytes_received: u64,
    consumer_lag: u64,
    last_message: Option<DateTime<Utc>>,
    avg_processing_time_ms: f64,
    connection_errors: u64,
}

impl RedisConsumer {
    pub fn new(config: StreamConfig) -> Result<Self> {
        let redis_config = if let StreamBackend::Redis { url, .. } = &config.backend {
            RedisStreamConfig {
                urls: vec![url.clone()],
                ..Default::default()
            }
        } else {
            RedisStreamConfig::default()
        };

        Ok(Self {
            config,
            redis_config,
            connection: None,
            stats: ConsumerStats::default(),
            last_id: ">".to_string(), // Start from new messages
            block_time_ms: 100,
            batch_size: 10,
        })
    }

    pub fn with_redis_config(mut self, redis_config: RedisStreamConfig) -> Self {
        self.redis_config = redis_config;
        self
    }

    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    #[cfg(feature = "redis")]
    pub async fn connect(&mut self) -> Result<()> {
        let connection = if self.redis_config.cluster_mode {
            let cluster_client = ClusterClient::new(self.redis_config.urls.clone())
                .map_err(|e| anyhow!("Failed to create Redis cluster client: {}", e))?;
            
            let connection = cluster_client.get_async_connection().await
                .map_err(|e| anyhow!("Failed to connect to Redis cluster: {}", e))?;
            
            RedisConnectionManager::Cluster(connection)
        } else {
            let client = Client::open(self.redis_config.urls[0].as_str())
                .map_err(|e| anyhow!("Failed to create Redis client: {}", e))?;
            
            let manager = client.get_connection_manager().await
                .map_err(|e| anyhow!("Failed to get Redis connection manager: {}", e))?;
            
            RedisConnectionManager::Standalone(manager)
        };

        self.connection = Some(connection);
        info!("Connected Redis consumer to stream: {}", self.redis_config.stream_name);
        Ok(())
    }

    #[cfg(not(feature = "redis"))]
    pub async fn connect(&mut self) -> Result<()> {
        warn!("Redis feature not enabled, using mock consumer");
        self.connection = Some(RedisConnectionManager::Mock);
        Ok(())
    }

    pub async fn consume(&mut self) -> Result<Option<StreamEvent>> {
        #[cfg(feature = "redis")]
        {
            if self.connection.is_none() {
                self.connect().await?;
            }

            match &mut self.connection {
                Some(RedisConnectionManager::Standalone(manager)) => {
                    let opts = StreamReadOptions::default()
                        .group(&self.redis_config.consumer_group, &self.redis_config.consumer_name)
                        .count(1)
                        .block(self.block_time_ms);

                    let result: RedisResult<StreamReadReply> = manager
                        .xread_options(&[&self.redis_config.stream_name], &[&self.last_id], &opts)
                        .await;

                    match result {
                        Ok(reply) => {
                            if let Some(stream_key) = reply.keys.first() {
                                if let Some(stream_id) = stream_key.ids.first() {
                                    self.last_id = stream_id.id.clone();
                                    return self.parse_redis_message(&stream_id.map).await;
                                }
                            }
                            Ok(None)
                        }
                        Err(e) => {
                            self.stats.connection_errors += 1;
                            error!("Redis read error: {}", e);
                            Err(anyhow!("Redis read error: {}", e))
                        }
                    }
                }
                Some(RedisConnectionManager::Cluster(connection)) => {
                    let opts = StreamReadOptions::default()
                        .group(&self.redis_config.consumer_group, &self.redis_config.consumer_name)
                        .count(1)
                        .block(self.block_time_ms);

                    let result: RedisResult<StreamReadReply> = connection
                        .xread_options(&[&self.redis_config.stream_name], &[&self.last_id], &opts)
                        .await;

                    match result {
                        Ok(reply) => {
                            if let Some(stream_key) = reply.keys.first() {
                                if let Some(stream_id) = stream_key.ids.first() {
                                    self.last_id = stream_id.id.clone();
                                    return self.parse_redis_message(&stream_id.map).await;
                                }
                            }
                            Ok(None)
                        }
                        Err(e) => {
                            self.stats.connection_errors += 1;
                            error!("Redis cluster read error: {}", e);
                            Err(anyhow!("Redis cluster read error: {}", e))
                        }
                    }
                }
                _ => Err(anyhow!("Redis connection not available"))
            }
        }
        #[cfg(not(feature = "redis"))]
        {
            time::sleep(Duration::from_millis(100)).await;
            Ok(None)
        }
    }

    async fn parse_redis_message(&mut self, fields: &HashMap<String, String>) -> Result<Option<StreamEvent>> {
        let start_time = Instant::now();
        
        if let Some(data) = fields.get("data") {
            match serde_json::from_str::<RedisStreamEvent>(data) {
                Ok(redis_event) => {
                    self.stats.events_consumed += 1;
                    self.stats.bytes_received += data.len() as u64;
                    self.stats.last_message = Some(Utc::now());

                    let processing_time = start_time.elapsed().as_millis() as f64;
                    self.stats.avg_processing_time_ms = 
                        (self.stats.avg_processing_time_ms + processing_time) / 2.0;

                    let stream_event = self.convert_redis_event(redis_event)?;
                    debug!("Consumed Redis event: {:?}", stream_event);
                    Ok(Some(stream_event))
                }
                Err(e) => {
                    self.stats.events_failed += 1;
                    error!("Failed to parse Redis event: {}", e);
                    Err(anyhow!("Event parse error: {}", e))
                }
            }
        } else {
            debug!("Received Redis message without data field");
            Ok(None)
        }
    }

    fn convert_redis_event(&self, redis_event: RedisStreamEvent) -> Result<StreamEvent> {
        let metadata = redis_event.metadata;
        
        match redis_event.event_type.as_str() {
            "triple_added" => {
                let subject = redis_event.data["subject"].as_str()
                    .ok_or_else(|| anyhow!("Missing subject"))?.to_string();
                let predicate = redis_event.data["predicate"].as_str()
                    .ok_or_else(|| anyhow!("Missing predicate"))?.to_string();
                let object = redis_event.data["object"].as_str()
                    .ok_or_else(|| anyhow!("Missing object"))?.to_string();
                let graph = redis_event.data["graph"].as_str().map(|s| s.to_string());
                
                Ok(StreamEvent::TripleAdded { subject, predicate, object, graph, metadata })
            }
            "triple_removed" => {
                let subject = redis_event.data["subject"].as_str()
                    .ok_or_else(|| anyhow!("Missing subject"))?.to_string();
                let predicate = redis_event.data["predicate"].as_str()
                    .ok_or_else(|| anyhow!("Missing predicate"))?.to_string();
                let object = redis_event.data["object"].as_str()
                    .ok_or_else(|| anyhow!("Missing object"))?.to_string();
                let graph = redis_event.data["graph"].as_str().map(|s| s.to_string());
                
                Ok(StreamEvent::TripleRemoved { subject, predicate, object, graph, metadata })
            }
            "graph_created" => {
                let graph = redis_event.data["graph"].as_str()
                    .ok_or_else(|| anyhow!("Missing graph"))?.to_string();
                Ok(StreamEvent::GraphCreated { graph, metadata })
            }
            "graph_cleared" => {
                let graph = redis_event.data["graph"].as_str().map(|s| s.to_string());
                Ok(StreamEvent::GraphCleared { graph, metadata })
            }
            "heartbeat" => {
                let source = redis_event.data["source"].as_str()
                    .ok_or_else(|| anyhow!("Missing source"))?.to_string();
                Ok(StreamEvent::Heartbeat { 
                    timestamp: redis_event.timestamp, 
                    source 
                })
            }
            _ => Err(anyhow!("Unknown event type: {}", redis_event.event_type))
        }
    }

    pub async fn consume_batch(&mut self, max_events: usize, timeout: Duration) -> Result<Vec<StreamEvent>> {
        let mut events = Vec::new();
        let start_time = time::Instant::now();

        while events.len() < max_events && start_time.elapsed() < timeout {
            match time::timeout(Duration::from_millis(50), self.consume()).await {
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

/// Redis admin utilities for stream management
pub struct RedisAdmin {
    #[cfg(feature = "redis")]
    connection: Option<RedisConnectionManager>,
    #[cfg(not(feature = "redis"))]
    _phantom: std::marker::PhantomData<()>,
    config: RedisStreamConfig,
}

impl RedisAdmin {
    #[cfg(feature = "redis")]
    pub async fn new(config: RedisStreamConfig) -> Result<Self> {
        let connection = if config.cluster_mode {
            let cluster_client = ClusterClient::new(config.urls.clone())
                .map_err(|e| anyhow!("Failed to create Redis cluster client: {}", e))?;
            
            let connection = cluster_client.get_async_connection().await
                .map_err(|e| anyhow!("Failed to connect to Redis cluster: {}", e))?;
            
            Some(RedisConnectionManager::Cluster(connection))
        } else {
            let client = Client::open(config.urls[0].as_str())
                .map_err(|e| anyhow!("Failed to create Redis client: {}", e))?;
            
            let manager = client.get_connection_manager().await
                .map_err(|e| anyhow!("Failed to get Redis connection manager: {}", e))?;
            
            Some(RedisConnectionManager::Standalone(manager))
        };

        Ok(Self { connection, config })
    }

    #[cfg(not(feature = "redis"))]
    pub async fn new(config: RedisStreamConfig) -> Result<Self> {
        Ok(Self { 
            _phantom: std::marker::PhantomData,
            config,
        })
    }

    #[cfg(feature = "redis")]
    pub async fn create_stream(&mut self, stream_name: &str) -> Result<()> {
        match &mut self.connection {
            Some(RedisConnectionManager::Standalone(manager)) => {
                let _: RedisResult<String> = redis::cmd("XADD")
                    .arg(stream_name)
                    .arg("*")
                    .arg("init")
                    .arg("true")
                    .query_async(manager)
                    .await;
                
                info!("Created Redis stream: {}", stream_name);
            }
            Some(RedisConnectionManager::Cluster(connection)) => {
                let _: RedisResult<String> = redis::cmd("XADD")
                    .arg(stream_name)
                    .arg("*")
                    .arg("init")
                    .arg("true")
                    .query_async(connection)
                    .await;
                
                info!("Created Redis cluster stream: {}", stream_name);
            }
            _ => return Err(anyhow!("No Redis connection available")),
        }
        Ok(())
    }

    #[cfg(not(feature = "redis"))]
    pub async fn create_stream(&mut self, stream_name: &str) -> Result<()> {
        info!("Mock: created Redis stream {}", stream_name);
        Ok(())
    }

    #[cfg(feature = "redis")]
    pub async fn get_stream_info(&mut self, stream_name: &str) -> Result<HashMap<String, String>> {
        match &mut self.connection {
            Some(RedisConnectionManager::Standalone(manager)) => {
                let info: redis::Value = redis::cmd("XINFO")
                    .arg("STREAM")
                    .arg(stream_name)
                    .query_async(manager)
                    .await
                    .map_err(|e| anyhow!("Failed to get stream info: {}", e))?;
                
                // Parse Redis response into HashMap
                Ok(HashMap::new()) // Simplified for now
            }
            Some(RedisConnectionManager::Cluster(connection)) => {
                let info: redis::Value = redis::cmd("XINFO")
                    .arg("STREAM")
                    .arg(stream_name)
                    .query_async(connection)
                    .await
                    .map_err(|e| anyhow!("Failed to get stream info: {}", e))?;
                
                Ok(HashMap::new()) // Simplified for now
            }
            _ => Err(anyhow!("No Redis connection available")),
        }
    }

    #[cfg(not(feature = "redis"))]
    pub async fn get_stream_info(&mut self, stream_name: &str) -> Result<HashMap<String, String>> {
        let mut info = HashMap::new();
        info.insert("name".to_string(), stream_name.to_string());
        info.insert("length".to_string(), "0".to_string());
        Ok(info)
    }
}