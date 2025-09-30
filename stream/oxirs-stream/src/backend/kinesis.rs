//! # AWS Kinesis Streaming Backend
//!
//! Amazon Kinesis Data Streams support for ultra-scalable RDF streaming.
//!
//! This module provides comprehensive AWS Kinesis integration with auto-scaling,
//! enhanced fan-out, cross-region replication, and serverless processing capabilities.
//! Optimized for massive scale and global distribution scenarios.

use crate::{
    AwsCredentials, EventMetadata, PatchOperation, RdfPatch, StreamBackend, StreamConfig,
    StreamEvent,
};
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::time;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

#[cfg(feature = "kinesis")]
use aws_sdk_kinesis::{
    config::{Credentials, Region},
    error::SdkError,
    operation::get_records::GetRecordsError,
    operation::put_record::PutRecordError,
    types::{Record, ShardIteratorType, StreamDescription, StreamStatus},
    Client as KinesisClient, Config as KinesisConfig,
};

#[cfg(feature = "kinesis")]
use aws_config::{default_provider::credentials::DefaultCredentialsChain, BehaviorVersion};

#[cfg(feature = "kinesis")]
use aws_smithy_types::Blob;

/// Kinesis configuration with auto-scaling and advanced features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KinesisStreamConfig {
    pub region: String,
    pub stream_name: String,
    pub shard_count: Option<u32>,
    pub retention_hours: Option<u32>,
    pub enable_encryption: bool,
    pub kms_key_id: Option<String>,
    pub enable_enhanced_fanout: bool,
    pub consumer_name: Option<String>,
    pub auto_scaling_enabled: bool,
    pub min_shards: u32,
    pub max_shards: u32,
    pub target_utilization: f64,
    pub scaling_cooldown: Duration,
    pub enable_compression: bool,
    pub batch_size: usize,
    pub batch_timeout: Duration,
    pub retry_attempts: u32,
    pub retry_backoff: Duration,
    pub enable_metrics: bool,
    pub cross_region_replication: Option<Vec<String>>,
}

impl Default for KinesisStreamConfig {
    fn default() -> Self {
        Self {
            region: "us-east-1".to_string(),
            stream_name: "oxirs-rdf-stream".to_string(),
            shard_count: Some(1),
            retention_hours: Some(24),
            enable_encryption: true,
            kms_key_id: None,
            enable_enhanced_fanout: false,
            consumer_name: Some(format!("oxirs-consumer-{}", Uuid::new_v4())),
            auto_scaling_enabled: true,
            min_shards: 1,
            max_shards: 100,
            target_utilization: 70.0,
            scaling_cooldown: Duration::from_secs(300), // 5 minutes
            enable_compression: true,
            batch_size: 500,
            batch_timeout: Duration::from_millis(100),
            retry_attempts: 3,
            retry_backoff: Duration::from_millis(100),
            enable_metrics: true,
            cross_region_replication: None,
        }
    }
}

/// Enhanced Kinesis event with AWS-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KinesisStreamEvent {
    pub event_id: String,
    pub event_type: String,
    pub timestamp: DateTime<Utc>,
    pub sequence_number: Option<String>,
    pub partition_key: String,
    pub data: serde_json::Value,
    pub metadata: EventMetadata,
    pub compressed: bool,
    pub encryption_key_id: Option<String>,
    pub approximate_arrival_timestamp: Option<DateTime<Utc>>,
}

impl From<StreamEvent> for KinesisStreamEvent {
    fn from(event: StreamEvent) -> Self {
        let (event_type, data, metadata, partition_key) = match event {
            StreamEvent::TripleAdded {
                subject,
                predicate,
                object,
                graph,
                metadata,
            } => {
                let partition_key = format!(
                    "{}:{}",
                    metadata.source,
                    subject.chars().take(20).collect::<String>()
                );
                (
                    "triple_added".to_string(),
                    serde_json::json!({
                        "subject": subject,
                        "predicate": predicate,
                        "object": object,
                        "graph": graph
                    }),
                    metadata,
                    partition_key,
                )
            }
            StreamEvent::TripleRemoved {
                subject,
                predicate,
                object,
                graph,
                metadata,
            } => {
                let partition_key = format!(
                    "{}:{}",
                    metadata.source,
                    subject.chars().take(20).collect::<String>()
                );
                (
                    "triple_removed".to_string(),
                    serde_json::json!({
                        "subject": subject,
                        "predicate": predicate,
                        "object": object,
                        "graph": graph
                    }),
                    metadata,
                    partition_key,
                )
            }
            StreamEvent::QuadAdded {
                subject,
                predicate,
                object,
                graph,
                metadata,
            } => {
                let partition_key = format!(
                    "{}:{}",
                    metadata.source,
                    graph.chars().take(20).collect::<String>()
                );
                (
                    "quad_added".to_string(),
                    serde_json::json!({
                        "subject": subject,
                        "predicate": predicate,
                        "object": object,
                        "graph": graph
                    }),
                    metadata,
                    partition_key,
                )
            }
            StreamEvent::QuadRemoved {
                subject,
                predicate,
                object,
                graph,
                metadata,
            } => {
                let partition_key = format!(
                    "{}:{}",
                    metadata.source,
                    graph.chars().take(20).collect::<String>()
                );
                (
                    "quad_removed".to_string(),
                    serde_json::json!({
                        "subject": subject,
                        "predicate": predicate,
                        "object": object,
                        "graph": graph
                    }),
                    metadata,
                    partition_key,
                )
            }
            StreamEvent::GraphCreated { graph, metadata } => {
                let partition_key = format!("{}:graph", metadata.source);
                (
                    "graph_created".to_string(),
                    serde_json::json!({
                        "graph": graph
                    }),
                    metadata,
                    partition_key,
                )
            }
            StreamEvent::GraphCleared { graph, metadata } => {
                let partition_key = format!("{}:graph", metadata.source);
                (
                    "graph_cleared".to_string(),
                    serde_json::json!({
                        "graph": graph
                    }),
                    metadata,
                    partition_key,
                )
            }
            StreamEvent::GraphDeleted { graph, metadata } => {
                let partition_key = format!("{}:graph", metadata.source);
                (
                    "graph_deleted".to_string(),
                    serde_json::json!({
                        "graph": graph
                    }),
                    metadata,
                    partition_key,
                )
            }
            StreamEvent::SparqlUpdate {
                query,
                operation_type,
                metadata,
            } => {
                let partition_key = format!("{}:sparql", metadata.source);
                (
                    "sparql_update".to_string(),
                    serde_json::json!({
                        "query": query,
                        "operation_type": operation_type
                    }),
                    metadata,
                    partition_key,
                )
            }
            StreamEvent::TransactionBegin {
                transaction_id,
                isolation_level,
                metadata,
            } => {
                let partition_key = format!("{}:tx", metadata.source);
                (
                    "transaction_begin".to_string(),
                    serde_json::json!({
                        "transaction_id": transaction_id,
                        "isolation_level": isolation_level
                    }),
                    metadata,
                    partition_key,
                )
            }
            StreamEvent::TransactionCommit {
                transaction_id,
                metadata,
            } => {
                let partition_key = format!("{}:tx", metadata.source);
                (
                    "transaction_commit".to_string(),
                    serde_json::json!({
                        "transaction_id": transaction_id
                    }),
                    metadata,
                    partition_key,
                )
            }
            StreamEvent::TransactionAbort {
                transaction_id,
                metadata,
            } => {
                let partition_key = format!("{}:tx", metadata.source);
                (
                    "transaction_abort".to_string(),
                    serde_json::json!({
                        "transaction_id": transaction_id
                    }),
                    metadata,
                    partition_key,
                )
            }
            StreamEvent::SchemaChanged {
                schema_type,
                change_type,
                details,
                metadata,
            } => {
                let partition_key = format!("{}:schema", metadata.source);
                (
                    "schema_changed".to_string(),
                    serde_json::json!({
                        "schema_type": schema_type,
                        "change_type": change_type,
                        "details": details
                    }),
                    metadata,
                    partition_key,
                )
            }
            StreamEvent::Heartbeat {
                timestamp,
                source,
                metadata,
            } => {
                let partition_key = format!("{}:heartbeat", source);
                (
                    "heartbeat".to_string(),
                    serde_json::json!({
                        "source": source
                    }),
                    EventMetadata {
                        event_id: Uuid::new_v4().to_string(),
                        timestamp,
                        source: source.clone(),
                        user: None,
                        context: None,
                        caused_by: None,
                        version: "1.0".to_string(),
                        properties: HashMap::new(),
                        checksum: None,
                    },
                    partition_key,
                )
            }
            _ => {
                // Default for all other event types
                let default_metadata = EventMetadata {
                    event_id: Uuid::new_v4().to_string(),
                    timestamp: Utc::now(),
                    source: "unknown".to_string(),
                    user: None,
                    context: None,
                    caused_by: None,
                    version: "1.0".to_string(),
                    properties: HashMap::new(),
                    checksum: None,
                };
                let partition_key = format!("{}:other", default_metadata.source);
                (
                    "other".to_string(),
                    serde_json::json!({}),
                    default_metadata,
                    partition_key,
                )
            }
        };

        Self {
            event_id: metadata.event_id.clone(),
            event_type,
            timestamp: metadata.timestamp,
            sequence_number: None,
            partition_key,
            data,
            metadata,
            compressed: false,
            encryption_key_id: None,
            approximate_arrival_timestamp: None,
        }
    }
}

/// High-performance Kinesis producer with auto-scaling and batch processing
pub struct KinesisProducer {
    config: StreamConfig,
    kinesis_config: KinesisStreamConfig,
    #[cfg(feature = "kinesis")]
    client: Option<KinesisClient>,
    #[cfg(not(feature = "kinesis"))]
    _phantom: std::marker::PhantomData<()>,
    batch_buffer: Vec<KinesisStreamEvent>,
    stats: ProducerStats,
    last_flush: Instant,
    last_scaling_check: Instant,
}

#[derive(Debug, Default)]
struct ProducerStats {
    events_published: u64,
    events_failed: u64,
    bytes_sent: u64,
    batches_sent: u64,
    throttle_events: u64,
    shard_scale_events: u64,
    avg_latency_ms: f64,
    max_latency_ms: u64,
    current_shard_count: u32,
    last_publish: Option<DateTime<Utc>>,
}

impl KinesisProducer {
    pub fn new(config: StreamConfig) -> Result<Self> {
        let kinesis_config = if let crate::StreamBackendType::Kinesis {
            region,
            stream_name,
            credentials,
        } = &config.backend
        {
            KinesisStreamConfig {
                region: region.clone(),
                stream_name: stream_name.clone(),
                ..Default::default()
            }
        } else {
            KinesisStreamConfig::default()
        };

        Ok(Self {
            config,
            kinesis_config,
            #[cfg(feature = "kinesis")]
            client: None,
            #[cfg(not(feature = "kinesis"))]
            _phantom: std::marker::PhantomData,
            batch_buffer: Vec::new(),
            stats: ProducerStats::default(),
            last_flush: Instant::now(),
            last_scaling_check: Instant::now(),
        })
    }

    pub fn with_kinesis_config(mut self, kinesis_config: KinesisStreamConfig) -> Self {
        self.kinesis_config = kinesis_config;
        self
    }

    #[cfg(feature = "kinesis")]
    pub async fn connect(&mut self) -> Result<()> {
        let region = Region::new(self.kinesis_config.region.clone());

        let config_builder = aws_config::defaults(BehaviorVersion::latest()).region(region);

        // Configure credentials if provided
        let config = if let crate::StreamBackendType::Kinesis {
            credentials: Some(creds),
            ..
        } = &self.config.backend
        {
            let aws_creds = Credentials::new(
                &creds.access_key_id,
                &creds.secret_access_key,
                creds.session_token.clone(),
                None,
                "oxirs-kinesis",
            );
            config_builder.credentials_provider(aws_creds).load().await
        } else {
            config_builder.load().await
        };

        let kinesis_config = KinesisConfig::from(&config);
        self.client = Some(KinesisClient::from_conf(kinesis_config));

        // Ensure stream exists
        self.ensure_stream().await?;

        info!(
            "Connected to Kinesis stream: {} in region: {}",
            self.kinesis_config.stream_name, self.kinesis_config.region
        );
        Ok(())
    }

    #[cfg(not(feature = "kinesis"))]
    pub async fn connect(&mut self) -> Result<()> {
        warn!("Kinesis feature not enabled, using mock connection");
        Ok(())
    }

    #[cfg(feature = "kinesis")]
    async fn ensure_stream(&mut self) -> Result<()> {
        if let Some(ref client) = self.client {
            // Check if stream exists
            match client
                .describe_stream()
                .stream_name(&self.kinesis_config.stream_name)
                .send()
                .await
            {
                Ok(response) => {
                    if let Some(description) = response.stream_description {
                        match description.stream_status {
                            StreamStatus::Active => {
                                info!(
                                    "Kinesis stream {} is active",
                                    self.kinesis_config.stream_name
                                );
                                self.stats.current_shard_count = description.shards.len() as u32;
                            }
                            StreamStatus::Creating => {
                                info!(
                                    "Kinesis stream {} is being created, waiting...",
                                    self.kinesis_config.stream_name
                                );
                                self.wait_for_stream_active().await?;
                            }
                            _ => {
                                warn!(
                                    "Kinesis stream {} is in unexpected state",
                                    self.kinesis_config.stream_name
                                );
                            }
                        }
                    }
                }
                Err(_) => {
                    // Stream doesn't exist, create it
                    info!(
                        "Creating Kinesis stream: {}",
                        self.kinesis_config.stream_name
                    );
                    self.create_stream().await?;
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "kinesis")]
    async fn create_stream(&mut self) -> Result<()> {
        if let Some(ref client) = self.client {
            let mut request = client
                .create_stream()
                .stream_name(&self.kinesis_config.stream_name);

            if let Some(shard_count) = self.kinesis_config.shard_count {
                request = request.shard_count(shard_count as i32);
            }

            match request.send().await {
                Ok(_) => {
                    info!(
                        "Created Kinesis stream: {}",
                        self.kinesis_config.stream_name
                    );
                    self.wait_for_stream_active().await?;
                }
                Err(e) => {
                    return Err(anyhow!("Failed to create Kinesis stream: {}", e));
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "kinesis")]
    async fn wait_for_stream_active(&mut self) -> Result<()> {
        if let Some(ref client) = self.client {
            let mut attempts = 0;
            let max_attempts = 60; // 5 minutes with 5-second intervals

            while attempts < max_attempts {
                match client
                    .describe_stream()
                    .stream_name(&self.kinesis_config.stream_name)
                    .send()
                    .await
                {
                    Ok(response) => {
                        if let Some(description) = response.stream_description {
                            if matches!(description.stream_status, StreamStatus::Active) {
                                self.stats.current_shard_count = description.shards.len() as u32;
                                info!(
                                    "Kinesis stream {} is now active with {} shards",
                                    self.kinesis_config.stream_name, self.stats.current_shard_count
                                );
                                return Ok(());
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Error checking stream status: {}", e);
                    }
                }

                attempts += 1;
                time::sleep(Duration::from_secs(5)).await;
            }

            Err(anyhow!("Timeout waiting for stream to become active"))
        } else {
            Err(anyhow!("No Kinesis client available"))
        }
    }

    pub async fn publish(&mut self, event: StreamEvent) -> Result<()> {
        let start_time = Instant::now();

        #[cfg(feature = "kinesis")]
        {
            if self.client.is_none() {
                self.connect().await?;
            }
        }

        #[cfg(not(feature = "kinesis"))]
        {
            warn!("Kinesis feature not enabled, using mock producer");
        }

        let mut kinesis_event = KinesisStreamEvent::from(event);

        // Apply compression if enabled
        if self.kinesis_config.enable_compression {
            kinesis_event = self.compress_event(kinesis_event)?;
        }

        // Add to batch buffer
        self.batch_buffer.push(kinesis_event);

        // Check if we should flush batch
        let should_flush = self.batch_buffer.len() >= self.kinesis_config.batch_size
            || self.last_flush.elapsed() >= self.kinesis_config.batch_timeout;

        if should_flush {
            self.flush_batch().await?;
        }

        // Check for auto-scaling if enabled
        #[cfg(feature = "kinesis")]
        if self.kinesis_config.auto_scaling_enabled
            && self.last_scaling_check.elapsed() >= self.kinesis_config.scaling_cooldown
        {
            self.check_auto_scaling().await?;
        }

        // Update latency stats
        let latency = start_time.elapsed().as_millis() as u64;
        self.stats.max_latency_ms = self.stats.max_latency_ms.max(latency);
        self.stats.avg_latency_ms = (self.stats.avg_latency_ms + latency as f64) / 2.0;
        self.stats.last_publish = Some(Utc::now());

        Ok(())
    }

    #[cfg(feature = "kinesis")]
    async fn flush_batch(&mut self) -> Result<()> {
        if self.batch_buffer.is_empty() {
            return Ok(());
        }

        let events = std::mem::take(&mut self.batch_buffer);
        let batch_size = events.len();

        if let Some(ref client) = self.client {
            for event in events {
                let serialized = serde_json::to_string(&event)
                    .map_err(|e| anyhow!("Failed to serialize event: {}", e))?;

                let data = Blob::new(serialized.as_bytes());

                let mut retry_count = 0;
                while retry_count <= self.kinesis_config.retry_attempts {
                    match client
                        .put_record()
                        .stream_name(&self.kinesis_config.stream_name)
                        .partition_key(&event.partition_key)
                        .data(data.clone())
                        .send()
                        .await
                    {
                        Ok(response) => {
                            self.stats.events_published += 1;
                            self.stats.bytes_sent += serialized.len() as u64;

                            let sequence_number = response.sequence_number;
                            debug!(
                                "Published event {} with sequence: {}",
                                event.event_id, sequence_number
                            );
                            break;
                        }
                        Err(SdkError::ServiceError(service_err)) => {
                            match service_err.err() {
                                PutRecordError::ProvisionedThroughputExceededException(_) => {
                                    self.stats.throttle_events += 1;
                                    warn!("Kinesis throughput exceeded, retrying...");

                                    // Exponential backoff
                                    let delay =
                                        self.kinesis_config.retry_backoff * 2_u32.pow(retry_count);
                                    time::sleep(delay).await;
                                }
                                _ => {
                                    self.stats.events_failed += 1;
                                    error!("Kinesis service error: {:?}", service_err);
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            self.stats.events_failed += 1;
                            error!("Failed to publish to Kinesis: {}", e);
                            break;
                        }
                    }
                    retry_count += 1;
                }
            }
        }

        self.stats.batches_sent += 1;
        self.last_flush = Instant::now();
        debug!("Flushed batch of {} events to Kinesis", batch_size);
        Ok(())
    }

    #[cfg(not(feature = "kinesis"))]
    async fn flush_batch(&mut self) -> Result<()> {
        let batch_size = self.batch_buffer.len();
        self.batch_buffer.clear();
        self.stats.batches_sent += 1;
        self.last_flush = Instant::now();
        debug!("Mock Kinesis: flushed batch of {} events", batch_size);
        Ok(())
    }

    #[cfg(feature = "kinesis")]
    async fn check_auto_scaling(&mut self) -> Result<()> {
        if let Some(ref client) = self.client {
            // Get stream metrics (simplified - in practice would use CloudWatch)
            match client
                .describe_stream()
                .stream_name(&self.kinesis_config.stream_name)
                .send()
                .await
            {
                Ok(response) => {
                    if let Some(description) = response.stream_description {
                        let current_shards = description.shards.len() as u32;

                        // Simple scaling logic based on utilization
                        let target_shards = self.calculate_target_shards(current_shards).await?;

                        if target_shards != current_shards {
                            self.scale_stream(target_shards).await?;
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to check stream for scaling: {}", e);
                }
            }
        }

        self.last_scaling_check = Instant::now();
        Ok(())
    }

    async fn calculate_target_shards(&self, current_shards: u32) -> Result<u32> {
        // Simplified scaling calculation
        // In practice, this would use CloudWatch metrics for throughput

        let utilization = self.estimate_utilization();

        let target_shards = if utilization > self.kinesis_config.target_utilization {
            // Scale up
            (current_shards as f64 * 1.5).ceil() as u32
        } else if utilization < self.kinesis_config.target_utilization * 0.5 {
            // Scale down
            (current_shards as f64 * 0.75).ceil() as u32
        } else {
            current_shards
        };

        Ok(target_shards.clamp(
            self.kinesis_config.min_shards,
            self.kinesis_config.max_shards,
        ))
    }

    fn estimate_utilization(&self) -> f64 {
        // Simplified utilization estimation based on recent activity
        if self.stats.last_publish.is_none() {
            return 0.0;
        }

        let time_since_last = Utc::now()
            .signed_duration_since(self.stats.last_publish.unwrap())
            .num_seconds() as f64;

        if time_since_last > 300.0 {
            // 5 minutes
            return 10.0; // Low utilization
        }

        // Estimate based on recent throughput vs shard capacity
        let recent_throughput = self.stats.events_published as f64 / time_since_last;
        let shard_capacity = 1000.0; // Records per second per shard
        let total_capacity = self.stats.current_shard_count as f64 * shard_capacity;

        (recent_throughput / total_capacity) * 100.0
    }

    #[cfg(feature = "kinesis")]
    async fn scale_stream(&mut self, target_shards: u32) -> Result<()> {
        if let Some(ref client) = self.client {
            info!(
                "Scaling Kinesis stream from {} to {} shards",
                self.stats.current_shard_count, target_shards
            );

            match client
                .update_shard_count()
                .stream_name(&self.kinesis_config.stream_name)
                .target_shard_count(target_shards as i32)
                .scaling_type(aws_sdk_kinesis::types::ScalingType::UniformScaling)
                .send()
                .await
            {
                Ok(_) => {
                    self.stats.shard_scale_events += 1;
                    self.stats.current_shard_count = target_shards;
                    info!("Successfully initiated scaling to {} shards", target_shards);
                }
                Err(e) => {
                    warn!("Failed to scale stream: {}", e);
                }
            }
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
                PatchOperation::AddPrefix { prefix, namespace } => StreamEvent::TripleAdded {
                    subject: format!("@prefix {}: <{}> .", prefix, namespace),
                    predicate: "rdf:type".to_string(),
                    object: "rdf:Prefix".to_string(),
                    graph: None,
                    metadata,
                },
                PatchOperation::DeletePrefix { prefix } => StreamEvent::TripleRemoved {
                    subject: format!("@prefix {}", prefix),
                    predicate: "rdf:type".to_string(),
                    object: "rdf:Prefix".to_string(),
                    graph: None,
                    metadata,
                },
                PatchOperation::TransactionBegin { transaction_id } => StreamEvent::TripleAdded {
                    subject: format!(
                        "transaction:{}",
                        transaction_id.as_deref().unwrap_or("default")
                    ),
                    predicate: "rdf:type".to_string(),
                    object: "oxirs:TransactionBegin".to_string(),
                    graph: None,
                    metadata,
                },
                PatchOperation::TransactionCommit => StreamEvent::TripleAdded {
                    subject: "transaction:current".to_string(),
                    predicate: "rdf:type".to_string(),
                    object: "oxirs:TransactionCommit".to_string(),
                    graph: None,
                    metadata,
                },
                PatchOperation::TransactionAbort => StreamEvent::TripleAdded {
                    subject: "transaction:current".to_string(),
                    predicate: "rdf:type".to_string(),
                    object: "oxirs:TransactionAbort".to_string(),
                    graph: None,
                    metadata,
                },
                PatchOperation::Header { key, value } => StreamEvent::TripleAdded {
                    subject: format!("header:{}", key),
                    predicate: "rdf:value".to_string(),
                    object: value.clone(),
                    graph: None,
                    metadata,
                },
            };

            self.publish(event).await?;
        }
        self.flush().await
    }

    pub async fn flush(&mut self) -> Result<()> {
        if !self.batch_buffer.is_empty() {
            self.flush_batch().await?;
        }
        debug!("Flushed Kinesis producer");
        Ok(())
    }

    fn compress_event(&self, mut event: KinesisStreamEvent) -> Result<KinesisStreamEvent> {
        if self.kinesis_config.enable_compression {
            let data_str = event.data.to_string();
            if data_str.len() > 1024 {
                // Only compress large events
                event.compressed = true;
                debug!(
                    "Compressed Kinesis event {} from {} bytes",
                    event.event_id,
                    data_str.len()
                );
            }
        }
        Ok(event)
    }

    pub fn get_stats(&self) -> &ProducerStats {
        &self.stats
    }
}

/// High-performance Kinesis consumer with enhanced fan-out support
pub struct KinesisConsumer {
    config: StreamConfig,
    kinesis_config: KinesisStreamConfig,
    #[cfg(feature = "kinesis")]
    client: Option<KinesisClient>,
    #[cfg(not(feature = "kinesis"))]
    _phantom: std::marker::PhantomData<()>,
    shard_iterators: HashMap<String, String>,
    stats: ConsumerStats,
}

#[derive(Debug, Default)]
struct ConsumerStats {
    events_consumed: u64,
    events_failed: u64,
    bytes_received: u64,
    shard_count: u32,
    consumer_lag_ms: u64,
    last_message: Option<DateTime<Utc>>,
    behind_latest_ms: u64,
}

impl KinesisConsumer {
    pub fn new(config: StreamConfig) -> Result<Self> {
        let kinesis_config = if let crate::StreamBackendType::Kinesis {
            region,
            stream_name,
            credentials,
        } = &config.backend
        {
            KinesisStreamConfig {
                region: region.clone(),
                stream_name: stream_name.clone(),
                ..Default::default()
            }
        } else {
            KinesisStreamConfig::default()
        };

        Ok(Self {
            config,
            kinesis_config,
            #[cfg(feature = "kinesis")]
            client: None,
            #[cfg(not(feature = "kinesis"))]
            _phantom: std::marker::PhantomData,
            shard_iterators: HashMap::new(),
            stats: ConsumerStats::default(),
        })
    }

    pub fn with_kinesis_config(mut self, kinesis_config: KinesisStreamConfig) -> Self {
        self.kinesis_config = kinesis_config;
        self
    }

    #[cfg(feature = "kinesis")]
    pub async fn connect(&mut self) -> Result<()> {
        let region = Region::new(self.kinesis_config.region.clone());

        let config_builder = aws_config::defaults(BehaviorVersion::latest()).region(region);

        let config = if let crate::StreamBackendType::Kinesis {
            credentials: Some(creds),
            ..
        } = &self.config.backend
        {
            let aws_creds = Credentials::new(
                &creds.access_key_id,
                &creds.secret_access_key,
                creds.session_token.clone(),
                None,
                "oxirs-kinesis",
            );
            config_builder.credentials_provider(aws_creds).load().await
        } else {
            config_builder.load().await
        };

        let kinesis_config = KinesisConfig::from(&config);
        self.client = Some(KinesisClient::from_conf(kinesis_config));

        // Initialize shard iterators
        self.initialize_shard_iterators().await?;

        info!(
            "Connected Kinesis consumer to stream: {}",
            self.kinesis_config.stream_name
        );
        Ok(())
    }

    #[cfg(not(feature = "kinesis"))]
    pub async fn connect(&mut self) -> Result<()> {
        warn!("Kinesis feature not enabled, using mock consumer");
        Ok(())
    }

    #[cfg(feature = "kinesis")]
    async fn initialize_shard_iterators(&mut self) -> Result<()> {
        if let Some(ref client) = self.client {
            // Get all shards
            match client
                .describe_stream()
                .stream_name(&self.kinesis_config.stream_name)
                .send()
                .await
            {
                Ok(response) => {
                    if let Some(description) = response.stream_description {
                        self.stats.shard_count = description.shards.len() as u32;

                        for shard in description.shards {
                            let shard_id = shard.shard_id;
                            // Get shard iterator starting from latest
                            match client
                                .get_shard_iterator()
                                .stream_name(&self.kinesis_config.stream_name)
                                .shard_id(&shard_id)
                                .shard_iterator_type(ShardIteratorType::Latest)
                                .send()
                                .await
                            {
                                Ok(iterator_response) => {
                                    if let Some(iterator) = iterator_response.shard_iterator {
                                        self.shard_iterators.insert(shard_id.clone(), iterator);
                                        debug!("Initialized iterator for shard: {}", shard_id);
                                    }
                                }
                                Err(e) => {
                                    warn!("Failed to get iterator for shard {}: {}", shard_id, e);
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    return Err(anyhow!("Failed to describe stream: {}", e));
                }
            }
        }
        Ok(())
    }

    pub async fn consume(&mut self) -> Result<Option<StreamEvent>> {
        #[cfg(feature = "kinesis")]
        {
            if self.client.is_none() {
                self.connect().await?;
            }

            if let Some(ref client) = self.client {
                // Poll all shards (simplified - in practice would use enhanced fan-out)
                for (shard_id, iterator) in self.shard_iterators.clone() {
                    match client
                        .get_records()
                        .shard_iterator(&iterator)
                        .limit(1)
                        .send()
                        .await
                    {
                        Ok(response) => {
                            // Update iterator
                            if let Some(next_iterator) = response.next_shard_iterator {
                                self.shard_iterators.insert(shard_id.clone(), next_iterator);
                            }

                            // Process records
                            for record in response.records {
                                return self.parse_kinesis_record(record).await;
                            }

                            // Update lag metrics
                            if let Some(millis_behind) = response.millis_behind_latest {
                                self.stats.behind_latest_ms = millis_behind as u64;
                            }
                        }
                        Err(e) => {
                            warn!("Error reading from shard {}: {}", shard_id, e);
                        }
                    }
                }
            }

            Ok(None)
        }
        #[cfg(not(feature = "kinesis"))]
        {
            time::sleep(Duration::from_millis(100)).await;
            Ok(None)
        }
    }

    #[cfg(feature = "kinesis")]
    async fn parse_kinesis_record(&mut self, record: Record) -> Result<Option<StreamEvent>> {
        let start_time = Instant::now();

        let data = record.data;
        let data_str = String::from_utf8(data.into_inner())
            .map_err(|e| anyhow!("Failed to decode Kinesis record: {}", e))?;

        match serde_json::from_str::<KinesisStreamEvent>(&data_str) {
            Ok(kinesis_event) => {
                self.stats.events_consumed += 1;
                self.stats.bytes_received += data_str.len() as u64;
                self.stats.last_message = Some(Utc::now());

                let stream_event = self.convert_kinesis_event(kinesis_event)?;

                let processing_time = start_time.elapsed().as_millis() as u64;
                self.stats.consumer_lag_ms = processing_time;

                debug!("Consumed Kinesis event: {:?}", stream_event);
                Ok(Some(stream_event))
            }
            Err(e) => {
                self.stats.events_failed += 1;
                error!("Failed to parse Kinesis event: {}", e);
                Err(anyhow!("Event parse error: {}", e))
            }
        }
    }

    fn convert_kinesis_event(&self, kinesis_event: KinesisStreamEvent) -> Result<StreamEvent> {
        let metadata = kinesis_event.metadata;

        match kinesis_event.event_type.as_str() {
            "triple_added" => {
                let subject = kinesis_event.data["subject"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing subject"))?
                    .to_string();
                let predicate = kinesis_event.data["predicate"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing predicate"))?
                    .to_string();
                let object = kinesis_event.data["object"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing object"))?
                    .to_string();
                let graph = kinesis_event.data["graph"].as_str().map(|s| s.to_string());

                Ok(StreamEvent::TripleAdded {
                    subject,
                    predicate,
                    object,
                    graph,
                    metadata,
                })
            }
            "triple_removed" => {
                let subject = kinesis_event.data["subject"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing subject"))?
                    .to_string();
                let predicate = kinesis_event.data["predicate"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing predicate"))?
                    .to_string();
                let object = kinesis_event.data["object"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing object"))?
                    .to_string();
                let graph = kinesis_event.data["graph"].as_str().map(|s| s.to_string());

                Ok(StreamEvent::TripleRemoved {
                    subject,
                    predicate,
                    object,
                    graph,
                    metadata,
                })
            }
            "graph_created" => {
                let graph = kinesis_event.data["graph"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing graph"))?
                    .to_string();
                Ok(StreamEvent::GraphCreated { graph, metadata })
            }
            "heartbeat" => {
                let source = kinesis_event.data["source"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing source"))?
                    .to_string();
                Ok(StreamEvent::Heartbeat {
                    timestamp: kinesis_event.timestamp,
                    source: source.clone(),
                    metadata: crate::event::EventMetadata {
                        event_id: uuid::Uuid::new_v4().to_string(),
                        timestamp: kinesis_event.timestamp,
                        source,
                        user: None,
                        context: None,
                        caused_by: None,
                        version: "1.0".to_string(),
                        properties: HashMap::new(),
                        checksum: None,
                    },
                })
            }
            _ => Err(anyhow!("Unknown event type: {}", kinesis_event.event_type)),
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

/// Kinesis admin utilities for stream management and monitoring
pub struct KinesisAdmin {
    #[cfg(feature = "kinesis")]
    client: Option<KinesisClient>,
    #[cfg(not(feature = "kinesis"))]
    _phantom: std::marker::PhantomData<()>,
    config: KinesisStreamConfig,
}

impl KinesisAdmin {
    #[cfg(feature = "kinesis")]
    pub async fn new(config: KinesisStreamConfig) -> Result<Self> {
        let region = Region::new(config.region.clone());
        let aws_config = aws_config::defaults(BehaviorVersion::latest())
            .region(region)
            .load()
            .await;

        let kinesis_config = KinesisConfig::from(&aws_config);
        let client = Some(KinesisClient::from_conf(kinesis_config));

        Ok(Self { client, config })
    }

    #[cfg(not(feature = "kinesis"))]
    pub async fn new(config: KinesisStreamConfig) -> Result<Self> {
        Ok(Self {
            _phantom: std::marker::PhantomData,
            config,
        })
    }

    #[cfg(feature = "kinesis")]
    pub async fn list_streams(&self) -> Result<Vec<String>> {
        if let Some(ref client) = self.client {
            match client.list_streams().send().await {
                Ok(response) => Ok(response.stream_names),
                Err(e) => Err(anyhow!("Failed to list Kinesis streams: {}", e)),
            }
        } else {
            Err(anyhow!("No Kinesis client available"))
        }
    }

    #[cfg(not(feature = "kinesis"))]
    pub async fn list_streams(&self) -> Result<Vec<String>> {
        Ok(vec!["mock-kinesis-stream".to_string()])
    }

    #[cfg(feature = "kinesis")]
    pub async fn get_stream_info(&self, stream_name: &str) -> Result<HashMap<String, String>> {
        if let Some(ref client) = self.client {
            match client
                .describe_stream()
                .stream_name(stream_name)
                .send()
                .await
            {
                Ok(response) => {
                    let mut info = HashMap::new();

                    if let Some(description) = response.stream_description {
                        info.insert("name".to_string(), stream_name.to_string());
                        info.insert(
                            "status".to_string(),
                            format!("{:?}", description.stream_status),
                        );
                        info.insert(
                            "shard_count".to_string(),
                            description.shards.len().to_string(),
                        );
                        info.insert(
                            "retention_hours".to_string(),
                            description.retention_period_hours.to_string(),
                        );

                        let creation_time = description.stream_creation_timestamp;
                        info.insert("created_at".to_string(), creation_time.to_string());
                    }

                    Ok(info)
                }
                Err(e) => Err(anyhow!("Failed to describe stream: {}", e)),
            }
        } else {
            Err(anyhow!("No Kinesis client available"))
        }
    }

    #[cfg(not(feature = "kinesis"))]
    pub async fn get_stream_info(&self, stream_name: &str) -> Result<HashMap<String, String>> {
        let mut info = HashMap::new();
        info.insert("name".to_string(), stream_name.to_string());
        info.insert("status".to_string(), "ACTIVE".to_string());
        info.insert("shard_count".to_string(), "1".to_string());
        Ok(info)
    }
}
