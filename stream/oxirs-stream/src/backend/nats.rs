//! # NATS Streaming Backend
//!
//! NATS support for streaming RDF data.
//!
//! This module provides lightweight NATS integration for streaming
//! RDF updates with JetStream for persistence and delivery guarantees.

// Note: KafkaEvent would be imported from backend::kafka if needed
// For now, using custom NatsEventMessage
use crate::{EventMetadata, PatchOperation, RdfPatch, StreamBackend, StreamConfig, StreamEvent};
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use futures_util::{StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time;

use tracing::{debug, error, info, warn};
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use std::collections::HashMap;

#[cfg(feature = "nats")]
use ::time::OffsetDateTime;

#[cfg(feature = "nats")]
use async_nats::{
    jetstream::{self, consumer::PullConsumer, stream::Stream},
    Client, ConnectOptions,
};

/// NATS-specific configuration with advanced JetStream features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NatsConfig {
    pub url: String,
    pub cluster_urls: Option<Vec<String>>,
    pub stream_name: String,
    pub subject_prefix: String,
    pub max_age_seconds: u64,
    pub max_bytes: u64,
    pub replicas: usize,
    pub storage_type: NatsStorageType,
    pub retention_policy: NatsRetentionPolicy,
    pub max_msgs: i64,
    pub max_msg_size: i32,
    pub discard_policy: NatsDiscardPolicy,
    pub duplicate_window: Duration,
    pub consumer_config: NatsConsumerConfig,
    pub auth_config: Option<NatsAuthConfig>,
    pub tls_config: Option<NatsTlsConfig>,
}

/// NATS storage types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NatsStorageType {
    File,
    Memory,
}

/// NATS retention policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NatsRetentionPolicy {
    Limits,
    Interest,
    WorkQueue,
}

/// NATS discard policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NatsDiscardPolicy {
    Old,
    New,
}

/// NATS consumer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NatsConsumerConfig {
    pub name: String,
    pub description: String,
    pub deliver_policy: NatsDeliverPolicy,
    pub ack_policy: NatsAckPolicy,
    pub ack_wait: Duration,
    pub max_deliver: i64,
    pub replay_policy: NatsReplayPolicy,
    pub max_ack_pending: i64,
    pub max_waiting: i64,
    pub max_batch: i64,
    pub max_expires: Duration,
    pub flow_control: bool,
    pub heartbeat: Duration,
}

/// NATS deliver policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NatsDeliverPolicy {
    All,
    Last,
    New,
    ByStartSequence(u64),
    ByStartTime(DateTime<Utc>),
    LastPerSubject,
}

/// NATS acknowledgment policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NatsAckPolicy {
    None,
    All,
    Explicit,
}

/// NATS replay policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NatsReplayPolicy {
    Instant,
    Original,
}

/// NATS event message format
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
        let (event_type, data, metadata) = match event {
            StreamEvent::TripleAdded { subject, predicate, object, graph, metadata } => (
                "triple_added".to_string(),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }),
                Some(metadata),
            ),
            StreamEvent::TripleRemoved { subject, predicate, object, graph, metadata } => (
                "triple_removed".to_string(),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }),
                Some(metadata),
            ),
            StreamEvent::QuadAdded { subject, predicate, object, graph, metadata } => (
                "quad_added".to_string(),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }),
                Some(metadata),
            ),
            StreamEvent::QuadRemoved { subject, predicate, object, graph, metadata } => (
                "quad_removed".to_string(),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }),
                Some(metadata),
            ),
            StreamEvent::GraphCreated { graph, metadata } => (
                "graph_created".to_string(),
                serde_json::json!({
                    "graph": graph
                }),
                Some(metadata),
            ),
            StreamEvent::GraphCleared { graph, metadata } => (
                "graph_cleared".to_string(),
                serde_json::json!({
                    "graph": graph
                }),
                Some(metadata),
            ),
            StreamEvent::GraphDeleted { graph, metadata } => (
                "graph_deleted".to_string(),
                serde_json::json!({
                    "graph": graph
                }),
                Some(metadata),
            ),
            StreamEvent::SparqlUpdate { query, operation_type, metadata } => (
                "sparql_update".to_string(),
                serde_json::json!({
                    "query": query,
                    "operation_type": operation_type
                }),
                Some(metadata),
            ),
            StreamEvent::TransactionBegin { transaction_id, isolation_level, metadata } => (
                "transaction_begin".to_string(),
                serde_json::json!({
                    "transaction_id": transaction_id,
                    "isolation_level": isolation_level
                }),
                Some(metadata),
            ),
            StreamEvent::TransactionCommit { transaction_id, metadata } => (
                "transaction_commit".to_string(),
                serde_json::json!({
                    "transaction_id": transaction_id
                }),
                Some(metadata),
            ),
            StreamEvent::TransactionAbort { transaction_id, metadata } => (
                "transaction_abort".to_string(),
                serde_json::json!({
                    "transaction_id": transaction_id
                }),
                Some(metadata),
            ),
            StreamEvent::SchemaChanged { schema_type, change_type, details, metadata } => (
                "schema_changed".to_string(),
                serde_json::json!({
                    "schema_type": schema_type,
                    "change_type": change_type,
                    "details": details
                }),
                Some(metadata),
            ),
            StreamEvent::Heartbeat { timestamp, source } => (
                "heartbeat".to_string(),
                serde_json::json!({
                    "source": source,
                    "timestamp": timestamp
                }),
                None,
            ),
            // Catch-all for remaining variants
            _ => (
                "unknown_event".to_string(),
                serde_json::json!({}),
                None,
            ),
        };

        Self {
            event_id: uuid::Uuid::new_v4().to_string(),
            event_type,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
            data,
            metadata,
        }
    }
}

impl NatsEventMessage {
    fn to_stream_event(&self) -> Result<StreamEvent> {
        let metadata = self.metadata.clone().unwrap_or_default();
        
        let event = match self.event_type.as_str() {
            "triple_added" => {
                let subject = self.data["subject"].as_str()
                    .ok_or_else(|| anyhow!("Missing subject"))?
                    .to_string();
                let predicate = self.data["predicate"].as_str()
                    .ok_or_else(|| anyhow!("Missing predicate"))?
                    .to_string();
                let object = self.data["object"].as_str()
                    .ok_or_else(|| anyhow!("Missing object"))?
                    .to_string();
                let graph = self.data["graph"].as_str().map(|s| s.to_string());
                
                StreamEvent::TripleAdded { subject, predicate, object, graph, metadata }
            }
            "triple_removed" => {
                let subject = self.data["subject"].as_str()
                    .ok_or_else(|| anyhow!("Missing subject"))?
                    .to_string();
                let predicate = self.data["predicate"].as_str()
                    .ok_or_else(|| anyhow!("Missing predicate"))?
                    .to_string();
                let object = self.data["object"].as_str()
                    .ok_or_else(|| anyhow!("Missing object"))?
                    .to_string();
                let graph = self.data["graph"].as_str().map(|s| s.to_string());
                
                StreamEvent::TripleRemoved { subject, predicate, object, graph, metadata }
            }
            "graph_created" => {
                let graph = self.data["graph"].as_str()
                    .ok_or_else(|| anyhow!("Missing graph"))?
                    .to_string();
                StreamEvent::GraphCreated { graph, metadata }
            }
            "graph_cleared" => {
                let graph = self.data["graph"].as_str().map(|s| s.to_string());
                StreamEvent::GraphCleared { graph, metadata }
            }
            "graph_deleted" => {
                let graph = self.data["graph"].as_str()
                    .ok_or_else(|| anyhow!("Missing graph"))?
                    .to_string();
                StreamEvent::GraphDeleted { graph, metadata }
            }
            _ => return Err(anyhow!("Unknown event type: {}", self.event_type)),
        };
        
        Ok(event)
    }
}

/// NATS authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NatsAuthConfig {
    pub token: Option<String>,
    pub username: Option<String>,
    pub password: Option<String>,
    pub nkey: Option<String>,
    pub credentials_file: Option<String>,
    pub jwt: Option<String>,
}

/// NATS TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NatsTlsConfig {
    pub ca_file: Option<String>,
    pub cert_file: Option<String>,
    pub key_file: Option<String>,
    pub verify: bool,
    pub timeout: Duration,
}

impl Default for NatsConfig {
    fn default() -> Self {
        Self {
            url: "nats://localhost:4222".to_string(),
            cluster_urls: None,
            stream_name: "OXIRS_RDF".to_string(),
            subject_prefix: "oxirs.rdf".to_string(),
            max_age_seconds: 86400,        // 24 hours
            max_bytes: 1024 * 1024 * 1024, // 1GB
            replicas: 1,
            storage_type: NatsStorageType::File,
            retention_policy: NatsRetentionPolicy::Limits,
            max_msgs: 1_000_000,
            max_msg_size: 1024 * 1024, // 1MB
            discard_policy: NatsDiscardPolicy::Old,
            duplicate_window: Duration::from_secs(120),
            consumer_config: NatsConsumerConfig::default(),
            auth_config: None,
            tls_config: None,
        }
    }
}

impl Default for NatsConsumerConfig {
    fn default() -> Self {
        Self {
            name: "oxirs-consumer".to_string(),
            description: "OxiRS RDF Stream Consumer".to_string(),
            deliver_policy: NatsDeliverPolicy::All,
            ack_policy: NatsAckPolicy::Explicit,
            ack_wait: Duration::from_secs(30),
            max_deliver: 3,
            replay_policy: NatsReplayPolicy::Instant,
            max_ack_pending: 1000,
            max_waiting: 512,
            max_batch: 100,
            max_expires: Duration::from_secs(60),
            flow_control: true,
            heartbeat: Duration::from_secs(5),
        }
    }
}

/// NATS producer for RDF streaming with cluster support
pub struct NatsProducer {
    config: StreamConfig,
    nats_config: NatsConfig,
    #[cfg(feature = "nats")]
    client: Option<Client>,
    #[cfg(feature = "nats")]
    jetstream: Option<jetstream::Context>,
    #[cfg(not(feature = "nats"))]
    _phantom: std::marker::PhantomData<()>,
    stats: Arc<RwLock<ProducerStats>>,
    publish_semaphore: Arc<Semaphore>,
    stream_metadata: Arc<RwLock<HashMap<String, StreamMetadata>>>,
    cluster_info: Arc<RwLock<ClusterInfo>>,
}

#[derive(Debug, Default, Clone)]
struct ProducerStats {
    events_published: u64,
    events_failed: u64,
    bytes_sent: u64,
    last_publish: Option<chrono::DateTime<chrono::Utc>>,
    stream_creates: u64,
    ack_timeouts: u64,
    delivery_errors: u64,
    avg_latency_ms: f64,
    max_latency_ms: u64,
}

/// Stream metadata for tracking stream state
#[derive(Debug, Clone)]
struct StreamMetadata {
    name: String,
    subjects: Vec<String>,
    message_count: u64,
    bytes_stored: u64,
    first_sequence: u64,
    last_sequence: u64,
    consumer_count: usize,
}

/// Cluster information for NATS
#[derive(Debug, Default, Clone)]
struct ClusterInfo {
    cluster_name: Option<String>,
    server_count: usize,
    leader_server: Option<String>,
    jetstream_enabled: bool,
    cluster_urls: Vec<String>,
}

impl NatsProducer {
    pub fn new(config: StreamConfig) -> Result<Self> {
        let nats_config = {
            #[cfg(feature = "nats")]
            {
                if let StreamBackend::Nats { url, .. } = &config.backend {
                    NatsConfig {
                        url: url.clone(),
                        ..Default::default()
                    }
                } else {
                    NatsConfig::default()
                }
            }
            #[cfg(not(feature = "nats"))]
            {
                NatsConfig::default()
            }
        };

        Ok(Self {
            config,
            nats_config,
            #[cfg(feature = "nats")]
            client: None,
            #[cfg(feature = "nats")]
            jetstream: None,
            #[cfg(not(feature = "nats"))]
            _phantom: std::marker::PhantomData,
            stats: Arc::new(RwLock::new(ProducerStats::default())),
        })
    }

    pub fn with_nats_config(mut self, nats_config: NatsConfig) -> Self {
        self.nats_config = nats_config;
        self
    }

    /// Apply authentication configuration
    #[cfg(feature = "nats")]
    fn apply_auth_config(&self, mut options: ConnectOptions, auth: &NatsAuthConfig) -> Result<ConnectOptions> {
        if let Some(ref token) = auth.token {
            options = options.token(token.clone());
        }
        if let (Some(ref username), Some(ref password)) = (&auth.username, &auth.password) {
            options = options.user_and_password(username.clone(), password.clone());
        }
        if let Some(ref nkey) = auth.nkey {
            options = options.nkey(nkey.clone())?;
        }
        if let Some(ref creds_file) = auth.credentials_file {
            options = options.credentials_file(creds_file)?;
        }
        Ok(options)
    }
    
    /// Apply TLS configuration
    #[cfg(feature = "nats")]
    fn apply_tls_config(&self, mut options: ConnectOptions, tls: &NatsTlsConfig) -> Result<ConnectOptions> {
        if tls.verify {
            options = options.require_tls();
        }
        // Additional TLS configuration would go here
        Ok(options)
    }
    
    #[cfg(feature = "nats")]
    pub async fn connect(&mut self) -> Result<()> {
        // Build connection options with cluster support
        let mut connect_options = ConnectOptions::new()
            .name("oxirs-nats-producer")
            .retry_on_initial_connect()
            .ping_interval(Duration::from_secs(10))
            .max_reconnects(10)
            .reconnect_delay_callback(|attempt| {
                std::cmp::min(attempt * 100, 5000).try_into().unwrap()
            });
        
        // Add authentication if configured
        if let Some(ref auth) = self.nats_config.auth_config {
            connect_options = self.apply_auth_config(connect_options, auth)?;
        }
        
        // Add TLS if configured
        if let Some(ref tls) = self.nats_config.tls_config {
            connect_options = self.apply_tls_config(connect_options, tls)?;
        }
        
        // Connect with cluster support
        let client = if let Some(ref cluster_urls) = self.nats_config.cluster_urls {
            let all_urls = std::iter::once(self.nats_config.url.clone())
                .chain(cluster_urls.iter().cloned())
                .collect::<Vec<_>>();  
            
            async_nats::connect_with_options(all_urls, connect_options)
                .await
                .map_err(|e| anyhow!("Failed to connect to NATS cluster: {}", e))?
        } else {
            async_nats::connect_with_options(self.nats_config.url.clone(), connect_options)
                .await
                .map_err(|e| anyhow!("Failed to connect to NATS: {}", e))?
        };

        let jetstream = jetstream::new(client.clone());

        // Create JetStream stream if it doesn't exist
        self.ensure_stream(&jetstream).await?;
        
        // Update cluster info
        if let Some(ref cluster_urls) = self.nats_config.cluster_urls {
            let mut cluster_info = self.cluster_info.write().await;
            cluster_info.cluster_urls = cluster_urls.clone();
            cluster_info.jetstream_enabled = true;
            cluster_info.server_count = cluster_urls.len() + 1;
        }

        self.client = Some(client);
        self.jetstream = Some(jetstream);

        info!("Connected to NATS at {} (cluster mode: {})", 
            self.nats_config.url, 
            self.nats_config.cluster_urls.is_some()
        );
        Ok(())
    }

    #[cfg(not(feature = "nats"))]
    pub async fn connect(&mut self) -> Result<()> {
        warn!("NATS feature not enabled, using mock connection");
        Ok(())
    }

    #[cfg(feature = "nats")]
    async fn ensure_stream(&self, jetstream: &jetstream::Context) -> Result<()> {
        let storage = match self.nats_config.storage_type {
            NatsStorageType::File => jetstream::stream::StorageType::File,
            NatsStorageType::Memory => jetstream::stream::StorageType::Memory,
        };

        let retention = match self.nats_config.retention_policy {
            NatsRetentionPolicy::Limits => jetstream::stream::RetentionPolicy::Limits,
            NatsRetentionPolicy::Interest => jetstream::stream::RetentionPolicy::Interest,
            NatsRetentionPolicy::WorkQueue => jetstream::stream::RetentionPolicy::WorkQueue,
        };

        let discard = match self.nats_config.discard_policy {
            NatsDiscardPolicy::Old => jetstream::stream::DiscardPolicy::Old,
            NatsDiscardPolicy::New => jetstream::stream::DiscardPolicy::New,
        };

        let stream_config = jetstream::stream::Config {
            name: self.nats_config.stream_name.clone(),
            subjects: vec![format!("{}.*", self.nats_config.subject_prefix)],
            max_age: Duration::from_secs(self.nats_config.max_age_seconds),
            max_bytes: self.nats_config.max_bytes as i64,
            max_messages: self.nats_config.max_msgs,
            max_message_size: self.nats_config.max_msg_size,
            num_replicas: self.nats_config.replicas,
            storage,
            retention,
            discard,
            duplicate_window: self.nats_config.duplicate_window,
            ..Default::default()
        };

        match jetstream.get_or_create_stream(stream_config).await {
            Ok(_) => {
                info!(
                    "Got or created JetStream stream: {} with {} replicas",
                    self.nats_config.stream_name, self.nats_config.replicas
                );
            }
            Err(e) => {
                return Err(anyhow!("Failed to get or create JetStream stream: {}", e));
            }
        }

        Ok(())
    }

    pub async fn publish(&mut self, event: StreamEvent) -> Result<()> {
        let start_time = std::time::Instant::now();
        let nats_event = NatsEventMessage::from(event);
        let subject = format!(
            "{}.{}",
            self.nats_config.subject_prefix, nats_event.event_type
        );

        #[cfg(feature = "nats")]
        {
            if self.jetstream.is_none() {
                self.connect().await?;
            }

            if let Some(ref jetstream) = self.jetstream {
                let payload = serde_json::to_string(&nats_event)
                    .map_err(|e| anyhow!("Failed to serialize event: {}", e))?;

                let headers = async_nats::HeaderMap::default();
                // Add correlation ID header
                // headers.insert("correlation-id", kafka_event.correlation_id.as_str());

                // Acquire publish permit
                let _permit = self.publish_semaphore.acquire().await
                    .map_err(|_| anyhow!("Failed to acquire publish permit"))?;
                
                match jetstream
                    .publish_with_headers(subject.clone(), headers, payload.clone().into())
                    .await
                {
                    Ok(ack) => {
                        let latency = start_time.elapsed().as_millis() as u64;
                        let mut stats = self.stats.write().await;
                        stats.events_published += 1;
                        stats.bytes_sent += payload.len() as u64;
                        stats.last_publish = Some(chrono::Utc::now());
                        stats.max_latency_ms = stats.max_latency_ms.max(latency);
                        stats.avg_latency_ms = (stats.avg_latency_ms + latency as f64) / 2.0;
                        
                        // Update stream metadata
                        if let Ok(mut metadata) = self.stream_metadata.write().await.get_mut(&self.nats_config.stream_name) {
                            metadata.message_count += 1;
                            metadata.bytes_stored += payload.len() as u64;
                            metadata.last_sequence = ack.sequence;
                        }
                        
                        debug!("Published event to NATS: {} (seq: {})", nats_event.event_id, ack.sequence);
                    }
                    Err(e) => {
                        let mut stats = self.stats.write().await;
                        stats.events_failed += 1;
                        stats.delivery_errors += 1;
                        error!("Failed to publish to NATS: {}", e);
                        return Err(anyhow!("NATS publish failed: {}", e));
                    }
                }
            } else {
                return Err(anyhow!("NATS JetStream not initialized"));
            }
        }
        #[cfg(not(feature = "nats"))]
        {
            debug!(
                "Mock NATS publish: {} to {}",
                kafka_event.correlation_id, subject
            );
            self.stats.events_published += 1;
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
                source: "nats_patch".to_string(),
                user: None,
                context: Some(patch.id.clone()),
                caused_by: None,
                version: "1.0".to_string(),
                properties: std::collections::HashMap::new(),
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
            };
            self.publish(event).await?;
        }
        self.flush().await
    }

    pub async fn flush(&mut self) -> Result<()> {
        #[cfg(feature = "nats")]
        {
            if let Some(ref client) = self.client {
                client
                    .flush()
                    .await
                    .map_err(|e| anyhow!("Failed to flush NATS client: {}", e))?;
                debug!("Flushed NATS client");
            }
        }
        #[cfg(not(feature = "nats"))]
        {
            debug!("Mock NATS flush");
        }
        Ok(())
    }

    pub fn get_stats(&self) -> &ProducerStats {
        &self.stats
    }
}

/// NATS consumer for RDF streaming with advanced features
pub struct NatsConsumer {
    config: StreamConfig,
    nats_config: NatsConfig,
    #[cfg(feature = "nats")]
    client: Option<Client>,
    #[cfg(feature = "nats")]
    consumer: Option<PullConsumer>,
    #[cfg(not(feature = "nats"))]
    _phantom: std::marker::PhantomData<()>,
    stats: Arc<RwLock<ConsumerStats>>,
    message_buffer: Arc<RwLock<Vec<NatsEventMessage>>>,
    consumer_state: Arc<RwLock<ConsumerState>>,
    health_checker: Arc<RwLock<ConsumerHealthChecker>>,
}

#[derive(Debug, Default, Clone)]
struct ConsumerStats {
    events_consumed: u64,
    events_failed: u64,
    bytes_received: u64,
    last_message: Option<chrono::DateTime<chrono::Utc>>,
    ack_success: u64,
    ack_failed: u64,
    redeliveries: u64,
    avg_processing_time_ms: f64,
    max_processing_time_ms: u64,
    pending_messages: u64,
}

/// Consumer state tracking
#[derive(Debug, Clone)]
struct ConsumerState {
    is_paused: bool,
    last_sequence: u64,
    consumer_name: String,
    pending_acks: HashMap<u64, std::time::Instant>,
}

impl Default for ConsumerState {
    fn default() -> Self {
        Self {
            is_paused: false,
            last_sequence: 0,
            consumer_name: format!("oxirs-consumer-{}", uuid::Uuid::new_v4()),
            pending_acks: HashMap::new(),
        }
    }
}

/// Consumer health checker
#[derive(Debug, Default)]
struct ConsumerHealthChecker {
    consecutive_failures: u32,
    last_health_check: Option<std::time::Instant>,
    is_healthy: bool,
    error_types: HashMap<String, u32>,
}

impl NatsConsumer {
    pub fn new(config: StreamConfig) -> Result<Self> {
        let nats_config = {
            #[cfg(feature = "nats")]
            {
                if let StreamBackend::Nats { url, .. } = &config.backend {
                    NatsConfig {
                        url: url.clone(),
                        ..Default::default()
                    }
                } else {
                    NatsConfig::default()
                }
            }
            #[cfg(not(feature = "nats"))]
            {
                NatsConfig::default()
            }
        };

        Ok(Self {
            config,
            nats_config,
            #[cfg(feature = "nats")]
            client: None,
            #[cfg(feature = "nats")]
            consumer: None,
            #[cfg(not(feature = "nats"))]
            _phantom: std::marker::PhantomData,
            stats: Arc::new(RwLock::new(ConsumerStats::default())),
            message_buffer: Arc::new(RwLock::new(Vec::with_capacity(100))),
            consumer_state: Arc::new(RwLock::new(ConsumerState::default())),
            health_checker: Arc::new(RwLock::new(ConsumerHealthChecker::default())),
        })
    }

    pub fn with_nats_config(mut self, nats_config: NatsConfig) -> Self {
        self.nats_config = nats_config;
        self
    }

    #[cfg(feature = "nats")]
    pub async fn connect(&mut self) -> Result<()> {
        let client = async_nats::connect(&self.nats_config.url)
            .await
            .map_err(|e| anyhow!("Failed to connect to NATS: {}", e))?;

        let jetstream = jetstream::new(client.clone());

        // Get or create the stream first
        let stream = jetstream
            .get_or_create_stream(jetstream::stream::Config {
                name: self.nats_config.stream_name.clone(),
                subjects: vec![format!("{}.*", self.nats_config.subject_prefix)],
                max_age: std::time::Duration::from_secs(self.nats_config.max_age_seconds),
                max_bytes: self.nats_config.max_bytes as i64,
                num_replicas: self.nats_config.replicas,
                ..Default::default()
            })
            .await
            .map_err(|e| anyhow!("Failed to create NATS stream: {}", e))?;

        // Create consumer on the stream with advanced configuration
        let deliver_policy = match self.nats_config.consumer_config.deliver_policy {
            NatsDeliverPolicy::All => jetstream::consumer::DeliverPolicy::All,
            NatsDeliverPolicy::Last => jetstream::consumer::DeliverPolicy::Last,
            NatsDeliverPolicy::New => jetstream::consumer::DeliverPolicy::New,
            NatsDeliverPolicy::ByStartSequence(seq) => jetstream::consumer::DeliverPolicy::ByStartSequence { start_sequence: seq },
            NatsDeliverPolicy::ByStartTime(time) => {
                #[cfg(feature = "nats")]
                {
                    // Convert DateTime<Utc> to OffsetDateTime
                    let unix_timestamp = time.timestamp();
                    let nanoseconds = time.timestamp_subsec_nanos();
                    let offset_datetime = OffsetDateTime::from_unix_timestamp_nanos(
                        unix_timestamp as i128 * 1_000_000_000 + nanoseconds as i128
                    ).unwrap_or_else(|_| OffsetDateTime::UNIX_EPOCH);
                    
                    jetstream::consumer::DeliverPolicy::ByStartTime { start_time: offset_datetime }
                }
                #[cfg(not(feature = "nats"))]
                {
                    // Fallback when NATS is not enabled
                    jetstream::consumer::DeliverPolicy::New
                }
            },
            NatsDeliverPolicy::LastPerSubject => jetstream::consumer::DeliverPolicy::LastPerSubject,
        };

        let ack_policy = match self.nats_config.consumer_config.ack_policy {
            NatsAckPolicy::None => jetstream::consumer::AckPolicy::None,
            NatsAckPolicy::All => jetstream::consumer::AckPolicy::All,
            NatsAckPolicy::Explicit => jetstream::consumer::AckPolicy::Explicit,
        };

        let replay_policy = match self.nats_config.consumer_config.replay_policy {
            NatsReplayPolicy::Instant => jetstream::consumer::ReplayPolicy::Instant,
            NatsReplayPolicy::Original => jetstream::consumer::ReplayPolicy::Original,
        };

        let consumer_config = jetstream::consumer::pull::Config {
            name: Some(self.nats_config.consumer_config.name.clone()),
            durable_name: Some(self.nats_config.consumer_config.name.clone()),
            description: Some(self.nats_config.consumer_config.description.clone()),
            deliver_policy,
            ack_policy,
            ack_wait: self.nats_config.consumer_config.ack_wait,
            max_deliver: self.nats_config.consumer_config.max_deliver,
            filter_subject: format!("{}.*", self.nats_config.subject_prefix),
            replay_policy,
            max_ack_pending: self.nats_config.consumer_config.max_ack_pending,
            max_waiting: self.nats_config.consumer_config.max_waiting,
            max_batch: self.nats_config.consumer_config.max_batch,
            max_expires: self.nats_config.consumer_config.max_expires,
            // Note: flow_control and heartbeat fields may not be available in this async_nats version
            ..Default::default()
        };

        let consumer = stream
            .create_consumer(consumer_config)
            .await
            .map_err(|e| anyhow!("Failed to create NATS consumer: {}", e))?;

        self.client = Some(client);
        self.consumer = Some(consumer);

        info!(
            "Connected NATS consumer to stream: {}",
            self.nats_config.stream_name
        );
        Ok(())
    }

    #[cfg(not(feature = "nats"))]
    pub async fn connect(&mut self) -> Result<()> {
        warn!("NATS feature not enabled, using mock consumer");
        Ok(())
    }

    /// Pause consumer (useful for backpressure)
    pub async fn pause(&mut self) {
        let mut state = self.consumer_state.write().await;
        state.is_paused = true;
        info!("NATS consumer paused");
    }
    
    /// Resume consumer
    pub async fn resume(&mut self) {
        let mut state = self.consumer_state.write().await;
        state.is_paused = false;
        info!("NATS consumer resumed");
    }
    
    /// Get consumer health status
    pub async fn is_healthy(&self) -> bool {
        let health = self.health_checker.read().await;
        health.is_healthy && health.consecutive_failures < 5
    }
    
    pub async fn consume(&mut self) -> Result<Option<StreamEvent>> {
        #[cfg(feature = "nats")]
        {
            // Check if paused
            if self.consumer_state.read().await.is_paused {
                tokio::time::sleep(Duration::from_millis(100)).await;
                return Ok(None);
            }
            
            if self.consumer.is_none() {
                self.connect().await?;
            }

            if let Some(ref consumer) = self.consumer {
                match consumer
                    .fetch()
                    .max_messages(1)
                    .max_bytes(1024 * 1024)
                    .messages()
                    .await
                {
                    Ok(mut messages) => {
                        if let Some(message) = messages.next().await {
                            match message {
                                Ok(msg) => {
                                    let payload =
                                        String::from_utf8(msg.payload.to_vec()).map_err(|e| {
                                            anyhow!("Failed to decode message payload: {}", e)
                                        })?;

                                    match serde_json::from_str::<NatsEventMessage>(&payload) {
                                        Ok(nats_event) => {
                                            let processing_start = std::time::Instant::now();
                                            let mut stats = self.stats.write().await;
                                            stats.events_consumed += 1;
                                            stats.bytes_received += payload.len() as u64;
                                            stats.last_message = Some(chrono::Utc::now());
                                            
                                            // Track pending acknowledgments
                                            let mut state = self.consumer_state.write().await;
                                            state.pending_acks.insert(msg.info().stream_sequence, processing_start);
                                            state.last_sequence = msg.info().stream_sequence;

                                            // Acknowledge the message
                                            let ack_result = msg.ack().await;
                                            let processing_time = processing_start.elapsed();
                                            
                                            // Update stats and state
                                            let mut stats = self.stats.write().await;
                                            let mut state = self.consumer_state.write().await;
                                            state.pending_acks.remove(&msg.info().stream_sequence);
                                            
                                            if let Err(e) = ack_result {
                                                warn!("Failed to acknowledge NATS message: {}", e);
                                                stats.ack_failed += 1;
                                            } else {
                                                stats.ack_success += 1;
                                                let time_ms = processing_time.as_millis() as u64;
                                                stats.max_processing_time_ms = stats.max_processing_time_ms.max(time_ms);
                                                stats.avg_processing_time_ms = (stats.avg_processing_time_ms + time_ms as f64) / 2.0;
                                            }
                                            
                                            // Update health status
                                            let mut health = self.health_checker.write().await;
                                            health.consecutive_failures = 0;
                                            health.is_healthy = true;
                                            health.last_health_check = Some(std::time::Instant::now());

                                            match nats_event.to_stream_event() {
                                                Ok(stream_event) => {
                                                    debug!(
                                                        "Consumed NATS event: {:?}",
                                                        stream_event
                                                    );
                                                    Ok(Some(stream_event))
                                                }
                                                Err(e) => {
                                                    self.stats.events_failed += 1;
                                                    error!("Failed to convert NATS event: {}", e);
                                                    Err(anyhow!("Event conversion failed: {}", e))
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            let mut stats = self.stats.write().await;
                                            stats.events_failed += 1;
                                            
                                            let mut health = self.health_checker.write().await;
                                            health.consecutive_failures += 1;
                                            *health.error_types.entry("parse_error".to_string()).or_insert(0) += 1;
                                            
                                            error!("Failed to parse NATS message: {}", e);
                                            Err(anyhow!("JSON parse error: {}", e))
                                        }
                                    }
                                }
                                Err(e) => {
                                    error!("NATS message error: {}", e);
                                    Err(anyhow!("NATS message error: {}", e))
                                }
                            }
                        } else {
                            Ok(None)
                        }
                    }
                    Err(e) => {
                        error!("NATS fetch error: {}", e);
                        Err(anyhow!("NATS fetch error: {}", e))
                    }
                }
            } else {
                Err(anyhow!("NATS consumer not initialized"))
            }
        }
        #[cfg(not(feature = "nats"))]
        {
            // Mock consumer - return None to simulate no messages
            time::sleep(Duration::from_millis(100)).await;
            Ok(None)
        }
    }

    pub async fn consume_batch(
        &mut self,
        max_events: usize,
        timeout: Duration,
    ) -> Result<Vec<StreamEvent>> {
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

/// NATS admin utilities
pub struct NatsAdmin {
    #[cfg(feature = "nats")]
    client: Client,
    #[cfg(feature = "nats")]
    jetstream: jetstream::Context,
    #[cfg(not(feature = "nats"))]
    _phantom: std::marker::PhantomData<()>,
}

impl NatsAdmin {
    #[cfg(feature = "nats")]
    pub async fn new(url: &str) -> Result<Self> {
        let client = async_nats::connect(url)
            .await
            .map_err(|e| anyhow!("Failed to connect to NATS: {}", e))?;

        let jetstream = jetstream::new(client.clone());

        Ok(Self { client, jetstream })
    }

    #[cfg(not(feature = "nats"))]
    pub async fn new(_url: &str) -> Result<Self> {
        Ok(Self {
            _phantom: std::marker::PhantomData,
        })
    }

    #[cfg(feature = "nats")]
    pub async fn list_streams(&self) -> Result<Vec<String>> {
        let mut stream_names = Vec::new();
        let mut streams = self.jetstream.streams();

        while let Some(stream) = streams
            .try_next()
            .await
            .map_err(|e| anyhow!("Failed to list streams: {}", e))?
        {
            stream_names.push(stream.config.name);
        }

        Ok(stream_names)
    }

    #[cfg(not(feature = "nats"))]
    pub async fn list_streams(&self) -> Result<Vec<String>> {
        Ok(vec!["mock-stream".to_string()])
    }

    #[cfg(feature = "nats")]
    pub async fn create_stream(&self, config: jetstream::stream::Config) -> Result<()> {
        match self.jetstream.create_stream(config.clone()).await {
            Ok(_) => {
                info!("Successfully created NATS stream: {}", config.name);
                Ok(())
            }
            Err(e) => {
                if e.to_string().contains("already exists") {
                    debug!("NATS stream '{}' already exists", config.name);
                    Ok(())
                } else {
                    Err(anyhow!(
                        "Failed to create NATS stream '{}': {}",
                        config.name,
                        e
                    ))
                }
            }
        }
    }

    #[cfg(not(feature = "nats"))]
    pub async fn create_stream(&self, _config: ()) -> Result<()> {
        info!("Mock: created NATS stream");
        Ok(())
    }

    #[cfg(feature = "nats")]
    pub async fn delete_stream(&self, name: &str) -> Result<()> {
        self.jetstream
            .delete_stream(name)
            .await
            .map_err(|e| anyhow!("Failed to delete stream '{}': {}", name, e))?;

        info!("Deleted NATS stream: {}", name);
        Ok(())
    }

    #[cfg(not(feature = "nats"))]
    pub async fn delete_stream(&self, name: &str) -> Result<()> {
        info!("Mock: deleted NATS stream {}", name);
        Ok(())
    }
}
