//! NATS Producer Implementation
//!
//! This module contains the producer implementation for NATS streaming backend.

use super::config::*;
use super::message::NatsEventMessage;
use crate::error::{StreamError, StreamResult};
use crate::{EventMetadata, PatchOperation, RdfPatch, StreamBackend, StreamConfig, StreamEvent};
use anyhow::{anyhow, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, error, info, warn};

#[cfg(feature = "nats")]
use async_nats::{
    jetstream::{self, consumer::PullConsumer, stream::Stream},
    Client, ConnectOptions,
};

/// Producer statistics for monitoring
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProducerStats {
    pub events_published: u64,
    pub events_failed: u64,
    pub bytes_sent: u64,
    pub delivery_errors: u64,
    pub last_publish: Option<chrono::DateTime<Utc>>,
    pub max_latency_ms: u64,
    pub avg_latency_ms: f64,
}

/// Stream metadata information
#[derive(Debug, Clone, Default)]
pub struct StreamMetadata {
    pub message_count: u64,
    pub bytes_stored: u64,
    pub last_sequence: u64,
    pub consumer_count: u32,
}

/// Cluster information
#[derive(Debug, Clone, Default)]
pub struct ClusterInfo {
    pub cluster_urls: Vec<String>,
    pub jetstream_enabled: bool,
    pub server_count: usize,
}

/// NATS Producer for streaming RDF events
pub struct NatsProducer {
    pub config: StreamConfig,
    pub nats_config: NatsConfig,
    #[cfg(feature = "nats")]
    pub client: Option<Client>,
    #[cfg(feature = "nats")]
    pub jetstream: Option<jetstream::Context>,
    #[cfg(not(feature = "nats"))]
    pub _phantom: std::marker::PhantomData<()>,
    pub stats: Arc<RwLock<ProducerStats>>,
    pub publish_semaphore: Arc<Semaphore>,
    pub stream_metadata: Arc<RwLock<HashMap<String, StreamMetadata>>>,
    pub cluster_info: Arc<RwLock<ClusterInfo>>,
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
            publish_semaphore: Arc::new(Semaphore::new(1000)),
            stream_metadata: Arc::new(RwLock::new(HashMap::new())),
            cluster_info: Arc::new(RwLock::new(ClusterInfo::default())),
        })
    }

    pub fn with_nats_config(mut self, nats_config: NatsConfig) -> Self {
        self.nats_config = nats_config;
        self
    }

    /// Apply authentication configuration
    #[cfg(feature = "nats")]
    async fn apply_auth_config(
        &self,
        mut options: ConnectOptions,
        auth: &NatsAuthConfig,
    ) -> Result<ConnectOptions> {
        if let Some(ref token) = auth.token {
            options = options.token(token.clone());
        }
        if let (Some(ref username), Some(ref password)) = (&auth.username, &auth.password) {
            options = options.user_and_password(username.clone(), password.clone());
        }
        if let Some(ref nkey) = auth.nkey {
            options = options.nkey(nkey.clone());
        }
        if let Some(ref creds_file) = auth.creds_file {
            options = options.credentials_file(creds_file).await?;
        }
        Ok(options)
    }

    /// Apply TLS configuration
    #[cfg(feature = "nats")]
    fn apply_tls_config(
        &self,
        mut options: ConnectOptions,
        tls: &NatsTlsConfig,
    ) -> Result<ConnectOptions> {
        if tls.verify_cert {
            options = options.require_tls(true);
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
            .reconnect_delay_callback(|attempt| {
                Duration::from_millis(std::cmp::min(attempt * 100, 5000) as u64)
            });

        // Add authentication if configured
        if let Some(ref auth) = self.nats_config.auth_config {
            connect_options = self.apply_auth_config(connect_options, auth).await?;
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

            // Convert Vec<String> to comma-separated string for NATS
            let urls_str = all_urls.join(",");
            async_nats::connect_with_options(urls_str, connect_options)
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

        info!(
            "Connected to NATS at {} (cluster mode: {})",
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

                // Acquire publish permit
                let _permit = self
                    .publish_semaphore
                    .acquire()
                    .await
                    .map_err(|_| anyhow!("Failed to acquire publish permit"))?;

                match jetstream
                    .publish_with_headers(subject.clone(), headers, payload.clone().into())
                    .await
                {
                    Ok(_ack) => {
                        let latency = start_time.elapsed().as_millis() as u64;
                        let mut stats = self.stats.write().await;
                        stats.events_published += 1;
                        stats.bytes_sent += payload.len() as u64;
                        stats.last_publish = Some(chrono::Utc::now());
                        stats.max_latency_ms = stats.max_latency_ms.max(latency);
                        stats.avg_latency_ms = (stats.avg_latency_ms + latency as f64) / 2.0;

                        // Update stream metadata
                        if let Some(metadata) = self
                            .stream_metadata
                            .write()
                            .await
                            .get_mut(&self.nats_config.stream_name)
                        {
                            metadata.message_count += 1;
                            metadata.bytes_stored += payload.len() as u64;
                        }

                        debug!("Published event to NATS: {}", nats_event.event_id);
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
            debug!("Mock NATS publish: {} to {}", nats_event.event_id, subject);
            let mut stats = self.stats.write().await;
            stats.events_published += 1;
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
                PatchOperation::AddPrefix { prefix, namespace } => {
                    StreamEvent::SchemaDefinitionAdded {
                        schema_type: "prefix".to_string(),
                        schema_uri: namespace.clone(),
                        definition: format!("PREFIX {} <{}>", prefix, namespace),
                        metadata,
                    }
                }
                PatchOperation::DeletePrefix { prefix } => StreamEvent::SchemaDefinitionRemoved {
                    schema_type: "prefix".to_string(),
                    schema_uri: format!("prefix:{}", prefix),
                    metadata,
                },
                PatchOperation::TransactionBegin { transaction_id } => {
                    StreamEvent::TransactionBegin {
                        transaction_id: transaction_id
                            .clone()
                            .unwrap_or_else(|| "unknown".to_string()),
                        isolation_level: None,
                        metadata,
                    }
                }
                PatchOperation::TransactionCommit => {
                    StreamEvent::TransactionCommit {
                        transaction_id: "unknown".to_string(),
                        metadata,
                    }
                }
                PatchOperation::TransactionAbort => {
                    StreamEvent::TransactionAbort {
                        transaction_id: "unknown".to_string(),
                        metadata,
                    }
                }
                PatchOperation::Header { key, value } => {
                    StreamEvent::SchemaDefinitionAdded {
                        schema_type: "header".to_string(),
                        schema_uri: key.clone(),
                        definition: format!("HEADER {} {}", key, value),
                        metadata,
                    }
                }
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

    pub async fn get_stats(&self) -> ProducerStats {
        self.stats.read().await.clone()
    }

    /// Determine if message should be processed based on load balancing strategy
    fn should_process_message(strategy: &LoadBalancingStrategy, message_count: u64, _subject: &str) -> bool {
        match strategy {
            LoadBalancingStrategy::RoundRobin => message_count % 2 == 0, // Simple implementation
            LoadBalancingStrategy::Random => rand::random::<bool>(),
            LoadBalancingStrategy::LeastConnections => true, // Simplified
            LoadBalancingStrategy::WeightedRoundRobin(_weights) => true, // Simplified
            LoadBalancingStrategy::Consistent => true, // Simplified
        }
    }

    // Circuit breaker helper methods
    async fn is_circuit_open(&self, _config: &CircuitBreakerConfig) -> Result<bool> {
        // Simplified implementation - in production, you'd maintain circuit state
        Ok(false)
    }

    async fn record_circuit_success(&self, _config: &CircuitBreakerConfig) -> Result<()> {
        // Record success for circuit breaker
        Ok(())
    }

    async fn record_circuit_failure(&self, _config: &CircuitBreakerConfig) -> Result<()> {
        // Record failure for circuit breaker
        Ok(())
    }
}