//! NATS Producer Implementation
//!
//! This module contains the producer implementation for NATS streaming backend.

use super::circuit_breaker::{
    CircuitBreaker, CircuitBreakerConfig as ResilientCircuitBreakerConfig,
};
use super::config::*;
use super::message::NatsEventMessage;
use crate::{EventMetadata, PatchOperation, RdfPatch, StreamConfig, StreamEvent};
use anyhow::{anyhow, Result};
use chrono::Utc;
use scirs2_core::random::rng;
use scirs2_core::RngExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, error, info};

#[cfg(feature = "nats")]
use async_nats::{
    jetstream::{self},
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
    /// Resilience circuit breaker, lazily built from
    /// `nats_config.request_reply_config.circuit_breaker` when configured. When
    /// present it gates `publish()` and records success/failure so a configured
    /// circuit breaker is actually enforced instead of being ignored.
    pub circuit_breaker: Arc<RwLock<Option<Arc<CircuitBreaker>>>>,
}

impl NatsProducer {
    pub fn new(config: StreamConfig) -> Result<Self> {
        let nats_config = {
            #[cfg(feature = "nats")]
            {
                if let crate::StreamBackendType::Nats { url, .. } = &config.backend {
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
            circuit_breaker: Arc::new(RwLock::new(None)),
        })
    }

    /// Lazily obtain the circuit breaker if one is configured.
    async fn get_circuit_breaker(&self) -> Option<Arc<CircuitBreaker>> {
        let cb_config = self
            .nats_config
            .request_reply_config
            .as_ref()
            .and_then(|rr| rr.circuit_breaker.clone())?;

        {
            let guard = self.circuit_breaker.read().await;
            if let Some(cb) = guard.as_ref() {
                return Some(cb.clone());
            }
        }

        let mut guard = self.circuit_breaker.write().await;
        if let Some(cb) = guard.as_ref() {
            return Some(cb.clone());
        }
        // Map the request-reply circuit breaker config onto the resilient
        // circuit breaker state machine's configuration.
        let resilient_config = ResilientCircuitBreakerConfig {
            failure_threshold: cb_config.failure_threshold,
            success_threshold: 1,
            timeout_seconds: cb_config.recovery_timeout.as_secs(),
            half_open_max_calls: cb_config.half_open_max_calls,
            enable_adaptive_thresholds: false,
            enable_ml_prediction: false,
            window_size_seconds: 60,
            slow_call_threshold_ms: 1000,
        };
        let cb = Arc::new(CircuitBreaker::new(resilient_config));
        *guard = Some(cb.clone());
        Some(cb)
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

        // Gate the publish on the circuit breaker when configured. An open
        // circuit fails fast instead of hammering an unhealthy broker.
        let circuit_breaker = self.get_circuit_breaker().await;
        if let Some(cb) = &circuit_breaker {
            if !cb.allow_call().await {
                let mut stats = self.stats.write().await;
                stats.events_failed += 1;
                return Err(anyhow!("NATS circuit breaker is open; publish rejected"));
            }
        }

        // If queue groups are configured, route to the group subject selected by
        // its load balancing strategy instead of the default subject.
        let subject = match self.select_queue_group_subject(&subject).await {
            Some(routed) => routed,
            None => subject,
        };

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
                        if let Some(cb) = &circuit_breaker {
                            cb.record_success(latency).await;
                        }
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
                        if let Some(cb) = &circuit_breaker {
                            cb.record_failure("publish_error").await;
                        }
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
            if let Some(cb) = &circuit_breaker {
                cb.record_success(start_time.elapsed().as_millis() as u64)
                    .await;
            }
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
                PatchOperation::TransactionCommit => StreamEvent::TransactionCommit {
                    transaction_id: "unknown".to_string(),
                    metadata,
                },
                PatchOperation::TransactionAbort => StreamEvent::TransactionAbort {
                    transaction_id: "unknown".to_string(),
                    metadata,
                },
                PatchOperation::Header { key, value } => StreamEvent::SchemaDefinitionAdded {
                    schema_type: "header".to_string(),
                    schema_uri: key.clone(),
                    definition: format!("HEADER {} {}", key, value),
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

    pub async fn get_stats(&self) -> ProducerStats {
        self.stats.read().await.clone()
    }

    /// Select which queue-group subject a message should be routed to based on
    /// the group's load balancing strategy, consulting `nats_config.queue_groups`
    /// (previously configured but never read). Returns `None` when no queue
    /// groups are configured, in which case the default subject is used.
    async fn select_queue_group_subject(&self, base_subject: &str) -> Option<String> {
        let group = self.nats_config.queue_groups.first()?;
        if group.subjects.is_empty() {
            return None;
        }

        let message_count = self.stats.read().await.events_published;
        let index = match &group.load_balancing_strategy {
            LoadBalancingStrategy::RoundRobin => (message_count as usize) % group.subjects.len(),
            LoadBalancingStrategy::Random => {
                (rng().random::<u64>() as usize) % group.subjects.len()
            }
            LoadBalancingStrategy::WeightedRoundRobin(weights) if !weights.is_empty() => {
                // Weighted selection over the provided weights, mapped onto the
                // available subjects.
                let total: u64 = weights.iter().map(|w| *w as u64).sum();
                if total == 0 {
                    (message_count as usize) % group.subjects.len()
                } else {
                    let mut target = message_count % total;
                    let mut chosen = 0usize;
                    for (i, w) in weights.iter().enumerate() {
                        let w = *w as u64;
                        if target < w {
                            chosen = i % group.subjects.len();
                            break;
                        }
                        target -= w;
                    }
                    chosen
                }
            }
            // Consistent hashing keyed by the base subject.
            LoadBalancingStrategy::Consistent => {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                base_subject.hash(&mut hasher);
                (hasher.finish() as usize) % group.subjects.len()
            }
            // LeastConnections and empty-weight WeightedRoundRobin fall back to
            // round-robin (no per-connection counters are tracked here).
            _ => (message_count as usize) % group.subjects.len(),
        };

        group.subjects.get(index).cloned()
    }
}
