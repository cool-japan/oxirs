//! Event streaming integration for real-time data updates
//!
//! This module provides integration with popular streaming platforms:
//! - Apache Kafka for distributed event streaming
//! - NATS for lightweight messaging
//! - Event sourcing capabilities
//! - Change Data Capture (CDC)
//! - Real-time analytics pipelines

pub mod cdc;
pub mod kafka;
pub mod nats;
pub mod pipeline;

use crate::error::FusekiResult;
use async_trait::async_trait;
use oxirs_core::{Quad, Triple};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::{mpsc, RwLock};
use url::Url;

/// Streaming configuration
#[derive(Debug, Clone, Default)]
pub struct StreamingConfig {
    /// Enable Kafka integration
    pub kafka: Option<KafkaConfig>,
    /// Enable NATS integration
    pub nats: Option<NatsConfig>,
    /// CDC configuration
    pub cdc: CDCConfig,
    /// Pipeline configuration
    pub pipeline: PipelineConfig,
}

/// Kafka configuration
#[derive(Debug, Clone)]
pub struct KafkaConfig {
    /// Kafka bootstrap servers
    pub brokers: Vec<String>,
    /// Topic prefix for RDF events
    pub topic_prefix: String,
    /// Producer configuration
    pub producer: ProducerConfig,
    /// Consumer configuration
    pub consumer: ConsumerConfig,
    /// Enable transactional semantics
    pub enable_transactions: bool,
}

/// Kafka producer configuration
#[derive(Debug, Clone)]
pub struct ProducerConfig {
    /// Compression type (none, gzip, snappy, lz4, zstd)
    pub compression: String,
    /// Batch size in bytes
    pub batch_size: usize,
    /// Linger time before sending batch
    pub linger_ms: u64,
    /// Request timeout
    pub request_timeout_ms: u64,
}

impl Default for ProducerConfig {
    fn default() -> Self {
        Self {
            compression: "snappy".to_string(),
            batch_size: 16384,
            linger_ms: 10,
            request_timeout_ms: 30000,
        }
    }
}

/// Kafka consumer configuration
#[derive(Debug, Clone)]
pub struct ConsumerConfig {
    /// Consumer group ID
    pub group_id: String,
    /// Auto offset reset (earliest, latest)
    pub auto_offset_reset: String,
    /// Enable auto commit
    pub enable_auto_commit: bool,
    /// Max poll records
    pub max_poll_records: usize,
}

impl Default for ConsumerConfig {
    fn default() -> Self {
        Self {
            group_id: "oxirs-consumer".to_string(),
            auto_offset_reset: "latest".to_string(),
            enable_auto_commit: true,
            max_poll_records: 500,
        }
    }
}

/// NATS configuration
#[derive(Debug, Clone)]
pub struct NatsConfig {
    /// NATS server URLs
    pub servers: Vec<Url>,
    /// Subject prefix for RDF events
    pub subject_prefix: String,
    /// Enable JetStream for persistence
    pub jetstream: bool,
    /// Authentication configuration
    pub auth: Option<NatsAuth>,
}

/// NATS authentication
#[derive(Debug, Clone)]
pub enum NatsAuth {
    /// Username/password authentication
    UserPass { username: String, password: String },
    /// Token authentication
    Token(String),
    /// NKey authentication
    NKey { seed: String },
}

/// Change Data Capture configuration
#[derive(Debug, Clone)]
pub struct CDCConfig {
    /// Enable CDC
    pub enabled: bool,
    /// Capture INSERT operations
    pub capture_inserts: bool,
    /// Capture DELETE operations
    pub capture_deletes: bool,
    /// Capture UPDATE operations (as delete+insert)
    pub capture_updates: bool,
    /// Include metadata in events
    pub include_metadata: bool,
    /// Batch size for CDC events
    pub batch_size: usize,
}

impl Default for CDCConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            capture_inserts: true,
            capture_deletes: true,
            capture_updates: true,
            include_metadata: true,
            batch_size: 100,
        }
    }
}

/// Pipeline configuration for stream processing
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Enable stream processing pipelines
    pub enabled: bool,
    /// Window size for time-based aggregations
    pub window_size: Duration,
    /// Watermark delay for late events
    pub watermark_delay: Duration,
    /// Maximum out-of-order delay
    pub max_out_of_order: Duration,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            window_size: Duration::from_secs(60),
            watermark_delay: Duration::from_secs(10),
            max_out_of_order: Duration::from_secs(300),
        }
    }
}

/// RDF event types for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RDFEvent {
    /// Triple added
    TripleAdded {
        #[serde(with = "triple_serde")]
        triple: Triple,
        graph: Option<String>,
        timestamp: i64,
    },
    /// Triple removed
    TripleRemoved {
        #[serde(with = "triple_serde")]
        triple: Triple,
        graph: Option<String>,
        timestamp: i64,
    },
    /// Quad added
    QuadAdded {
        #[serde(with = "quad_serde")]
        quad: Quad,
        timestamp: i64,
    },
    /// Quad removed
    QuadRemoved {
        #[serde(with = "quad_serde")]
        quad: Quad,
        timestamp: i64,
    },
    /// Graph cleared
    GraphCleared { graph: String, timestamp: i64 },
    /// Transaction event
    Transaction {
        id: String,
        events: Vec<RDFEvent>,
        timestamp: i64,
    },
}

/// Serialization helpers for RDF types
mod triple_serde {
    use super::*;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(triple: &Triple, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Triple", 3)?;
        state.serialize_field("subject", &format!("{}", triple.subject()))?;
        state.serialize_field("predicate", &format!("{}", triple.predicate()))?;
        state.serialize_field("object", &format!("{}", triple.object()))?;
        state.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<Triple, D::Error>
    where
        D: Deserializer<'de>,
    {
        use oxirs_core::{BlankNode, Literal, NamedNode, Subject, Term};
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        #[derive(Deserialize)]
        struct TripleHelper {
            subject: String,
            predicate: String,
            object: String,
        }

        struct TripleVisitor;

        impl<'de> Visitor<'de> for TripleVisitor {
            type Value = Triple;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a Triple struct")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Triple, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut subject: Option<String> = None;
                let mut predicate: Option<String> = None;
                let mut object: Option<String> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "subject" => {
                            if subject.is_some() {
                                return Err(de::Error::duplicate_field("subject"));
                            }
                            subject = Some(map.next_value()?);
                        }
                        "predicate" => {
                            if predicate.is_some() {
                                return Err(de::Error::duplicate_field("predicate"));
                            }
                            predicate = Some(map.next_value()?);
                        }
                        "object" => {
                            if object.is_some() {
                                return Err(de::Error::duplicate_field("object"));
                            }
                            object = Some(map.next_value()?);
                        }
                        _ => {
                            let _: String = map.next_value()?;
                        }
                    }
                }

                let subject = subject.ok_or_else(|| de::Error::missing_field("subject"))?;
                let predicate = predicate.ok_or_else(|| de::Error::missing_field("predicate"))?;
                let object = object.ok_or_else(|| de::Error::missing_field("object"))?;

                // Parse subject
                let subject = if let Some(stripped) = subject.strip_prefix("_:") {
                    Subject::BlankNode(
                        BlankNode::new(stripped)
                            .map_err(|e| de::Error::custom(format!("Invalid blank node: {e}")))?,
                    )
                } else if subject.starts_with('<') && subject.ends_with('>') {
                    let iri = &subject[1..subject.len() - 1];
                    Subject::NamedNode(
                        NamedNode::new(iri)
                            .map_err(|e| de::Error::custom(format!("Invalid IRI: {e}")))?,
                    )
                } else {
                    return Err(de::Error::custom("Invalid subject format"));
                };

                // Parse predicate
                let predicate = if predicate.starts_with('<') && predicate.ends_with('>') {
                    let iri = &predicate[1..predicate.len() - 1];
                    NamedNode::new(iri)
                        .map_err(|e| de::Error::custom(format!("Invalid predicate IRI: {e}")))?
                } else {
                    return Err(de::Error::custom("Invalid predicate format"));
                };

                // Parse object
                let object = if let Some(stripped) = object.strip_prefix("_:") {
                    Term::BlankNode(
                        BlankNode::new(stripped)
                            .map_err(|e| de::Error::custom(format!("Invalid blank node: {e}")))?,
                    )
                } else if object.starts_with('<') && object.ends_with('>') {
                    let iri = &object[1..object.len() - 1];
                    Term::NamedNode(
                        NamedNode::new(iri)
                            .map_err(|e| de::Error::custom(format!("Invalid IRI: {e}")))?,
                    )
                } else if object.starts_with('"') {
                    // Parse literal (simplified - just treat as string for now)
                    let literal_value = &object[1..object.len() - 1];
                    Term::Literal(Literal::new_simple_literal(literal_value))
                } else {
                    return Err(de::Error::custom("Invalid object format"));
                };

                Ok(Triple::new(subject, predicate, object))
            }
        }

        deserializer.deserialize_struct(
            "Triple",
            &["subject", "predicate", "object"],
            TripleVisitor,
        )
    }
}

mod quad_serde {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(quad: &Quad, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Quad", 4)?;
        state.serialize_field("subject", &format!("{}", quad.subject()))?;
        state.serialize_field("predicate", &format!("{}", quad.predicate()))?;
        state.serialize_field("object", &format!("{}", quad.object()))?;
        state.serialize_field("graph_name", &format!("{}", quad.graph_name()))?;
        state.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<Quad, D::Error>
    where
        D: Deserializer<'de>,
    {
        use oxirs_core::{BlankNode, GraphName, Literal, NamedNode, Subject, Term};
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        struct QuadVisitor;

        impl<'de> Visitor<'de> for QuadVisitor {
            type Value = Quad;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a Quad struct")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Quad, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut subject: Option<String> = None;
                let mut predicate: Option<String> = None;
                let mut object: Option<String> = None;
                let mut graph_name: Option<String> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "subject" => {
                            if subject.is_some() {
                                return Err(de::Error::duplicate_field("subject"));
                            }
                            subject = Some(map.next_value()?);
                        }
                        "predicate" => {
                            if predicate.is_some() {
                                return Err(de::Error::duplicate_field("predicate"));
                            }
                            predicate = Some(map.next_value()?);
                        }
                        "object" => {
                            if object.is_some() {
                                return Err(de::Error::duplicate_field("object"));
                            }
                            object = Some(map.next_value()?);
                        }
                        "graph_name" => {
                            if graph_name.is_some() {
                                return Err(de::Error::duplicate_field("graph_name"));
                            }
                            graph_name = Some(map.next_value()?);
                        }
                        _ => {
                            let _: String = map.next_value()?;
                        }
                    }
                }

                let subject = subject.ok_or_else(|| de::Error::missing_field("subject"))?;
                let predicate = predicate.ok_or_else(|| de::Error::missing_field("predicate"))?;
                let object = object.ok_or_else(|| de::Error::missing_field("object"))?;
                let graph_name =
                    graph_name.ok_or_else(|| de::Error::missing_field("graph_name"))?;

                // Parse subject
                let subject = if let Some(stripped) = subject.strip_prefix("_:") {
                    Subject::BlankNode(
                        BlankNode::new(stripped)
                            .map_err(|e| de::Error::custom(format!("Invalid blank node: {e}")))?,
                    )
                } else if subject.starts_with('<') && subject.ends_with('>') {
                    let iri = &subject[1..subject.len() - 1];
                    Subject::NamedNode(
                        NamedNode::new(iri)
                            .map_err(|e| de::Error::custom(format!("Invalid IRI: {e}")))?,
                    )
                } else {
                    return Err(de::Error::custom("Invalid subject format"));
                };

                // Parse predicate
                let predicate = if predicate.starts_with('<') && predicate.ends_with('>') {
                    let iri = &predicate[1..predicate.len() - 1];
                    NamedNode::new(iri)
                        .map_err(|e| de::Error::custom(format!("Invalid predicate IRI: {e}")))?
                } else {
                    return Err(de::Error::custom("Invalid predicate format"));
                };

                // Parse object
                let object = if let Some(stripped) = object.strip_prefix("_:") {
                    Term::BlankNode(
                        BlankNode::new(stripped)
                            .map_err(|e| de::Error::custom(format!("Invalid blank node: {e}")))?,
                    )
                } else if object.starts_with('<') && object.ends_with('>') {
                    let iri = &object[1..object.len() - 1];
                    Term::NamedNode(
                        NamedNode::new(iri)
                            .map_err(|e| de::Error::custom(format!("Invalid IRI: {e}")))?,
                    )
                } else if object.starts_with('"') {
                    // Parse literal (simplified - just treat as string for now)
                    let literal_value = &object[1..object.len() - 1];
                    Term::Literal(Literal::new_simple_literal(literal_value))
                } else {
                    return Err(de::Error::custom("Invalid object format"));
                };

                // Parse graph name
                let graph_name = if graph_name.starts_with('<') && graph_name.ends_with('>') {
                    let iri = &graph_name[1..graph_name.len() - 1];
                    GraphName::NamedNode(
                        NamedNode::new(iri)
                            .map_err(|e| de::Error::custom(format!("Invalid graph IRI: {e}")))?,
                    )
                } else if let Some(stripped) = graph_name.strip_prefix("_:") {
                    GraphName::BlankNode(
                        BlankNode::new(stripped)
                            .map_err(|e| de::Error::custom(format!("Invalid blank node: {e}")))?,
                    )
                } else {
                    return Err(de::Error::custom("Invalid graph name format"));
                };

                Ok(Quad::new(subject, predicate, object, graph_name))
            }
        }

        deserializer.deserialize_struct(
            "Quad",
            &["subject", "predicate", "object", "graph_name"],
            QuadVisitor,
        )
    }
}

/// Stream producer trait for sending RDF events
#[async_trait]
pub trait StreamProducer: Send + Sync {
    /// Send a single event
    async fn send(&self, event: RDFEvent) -> FusekiResult<()>;

    /// Send a batch of events
    async fn send_batch(&self, events: Vec<RDFEvent>) -> FusekiResult<()>;

    /// Flush any pending events
    async fn flush(&self) -> FusekiResult<()>;
}

/// Stream consumer trait for receiving RDF events
#[async_trait]
pub trait StreamConsumer: Send + Sync {
    /// Subscribe to events
    async fn subscribe(&self, handler: Box<dyn EventHandler>) -> FusekiResult<()>;

    /// Unsubscribe from events
    async fn unsubscribe(&self) -> FusekiResult<()>;

    /// Commit processed offsets (for Kafka)
    async fn commit(&self) -> FusekiResult<()>;
}

/// Event handler for processing streamed events
#[async_trait]
pub trait EventHandler: Send + Sync {
    /// Handle an RDF event
    async fn handle(&self, event: RDFEvent) -> FusekiResult<()>;

    /// Handle errors
    async fn on_error(&self, error: Box<dyn std::error::Error + Send + Sync>) {
        tracing::error!("Event handler error: {}", error);
    }
}

/// Streaming manager for coordinating producers and consumers
pub struct StreamingManager {
    config: StreamingConfig,
    producers: Arc<RwLock<HashMap<String, Box<dyn StreamProducer>>>>,
    consumers: Arc<RwLock<HashMap<String, Box<dyn StreamConsumer>>>>,
    event_buffer: mpsc::Sender<RDFEvent>,
    event_receiver: Arc<RwLock<mpsc::Receiver<RDFEvent>>>,
}

impl StreamingManager {
    /// Create a new streaming manager
    pub fn new(config: StreamingConfig) -> Self {
        let (tx, rx) = mpsc::channel(10000);

        Self {
            config,
            producers: Arc::new(RwLock::new(HashMap::new())),
            consumers: Arc::new(RwLock::new(HashMap::new())),
            event_buffer: tx,
            event_receiver: Arc::new(RwLock::new(rx)),
        }
    }

    /// Initialize streaming connections
    pub async fn initialize(&self) -> FusekiResult<()> {
        // Initialize Kafka if configured
        if let Some(kafka_config) = &self.config.kafka {
            tracing::info!("Initializing Kafka streaming");
            let kafka_client_config: crate::streaming::kafka::KafkaConfig =
                kafka_config.clone().into();
            let producer =
                crate::streaming::kafka::KafkaProducer::new(kafka_client_config.clone()).await?;
            let consumer = crate::streaming::kafka::KafkaConsumer::new(kafka_client_config).await?;

            let mut producers = self.producers.write().await;
            let mut consumers = self.consumers.write().await;

            producers.insert("kafka".to_string(), Box::new(producer));
            consumers.insert("kafka".to_string(), Box::new(consumer));
        }

        // Initialize NATS if configured
        if let Some(nats_config) = &self.config.nats {
            tracing::info!("Initializing NATS streaming");
            let nats_client_config: crate::streaming::nats::NatsConfig = nats_config.clone().into();
            let producer =
                crate::streaming::nats::NatsProducer::new(nats_client_config.clone()).await?;
            let consumer = crate::streaming::nats::NatsConsumer::new(nats_client_config).await?;

            let mut producers = self.producers.write().await;
            let mut consumers = self.consumers.write().await;

            producers.insert("nats".to_string(), Box::new(producer));
            consumers.insert("nats".to_string(), Box::new(consumer));
        }

        // Start event processing loop
        self.start_event_processor().await;

        Ok(())
    }

    /// Send an RDF event to all configured streams
    pub async fn send_event(&self, event: RDFEvent) -> crate::error::Result<()> {
        // Buffer the event
        self.event_buffer.send(event.clone()).await.map_err(|_| {
            crate::error::FusekiError::Internal {
                message: "Event buffer full".to_string(),
            }
        })?;

        Ok(())
    }

    /// Start the event processing loop
    async fn start_event_processor(&self) {
        let receiver = self.event_receiver.clone();
        let producers = self.producers.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut batch = Vec::new();
            let mut interval = tokio::time::interval(Duration::from_millis(100));

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if !batch.is_empty() && batch.len() >= config.cdc.batch_size {
                            Self::send_batch(&producers, batch.clone()).await;
                            batch.clear();
                        }
                    }
                    event = async {
                        let mut rx = receiver.write().await;
                        rx.recv().await
                    } => {
                        if let Some(event) = event {
                            batch.push(event);

                            // Send immediately if batch is full
                            if batch.len() >= config.cdc.batch_size {
                                Self::send_batch(&producers, batch.clone()).await;
                                batch.clear();
                            }
                        } else {
                            // Channel closed, send remaining batch
                            if !batch.is_empty() {
                                Self::send_batch(&producers, batch.clone()).await;
                            }
                            break;
                        }
                    }
                }
            }
        });
    }

    /// Send a batch of events to all producers
    async fn send_batch(
        producers: &Arc<RwLock<HashMap<String, Box<dyn StreamProducer>>>>,
        batch: Vec<RDFEvent>,
    ) {
        let producers = producers.read().await;

        for (name, producer) in producers.iter() {
            if let Err(e) = producer.send_batch(batch.clone()).await {
                tracing::error!("Failed to send batch to {}: {}", name, e);
            }
        }
    }

    /// Shutdown streaming connections
    pub async fn shutdown(&self) -> crate::error::Result<()> {
        // Flush all producers
        let producers = self.producers.read().await;
        for (name, producer) in producers.iter() {
            if let Err(e) = producer.flush().await {
                tracing::error!("Failed to flush {}: {}", name, e);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert!(config.kafka.is_none());
        assert!(config.nats.is_none());
        assert!(config.cdc.enabled);
    }

    #[test]
    fn test_rdf_event_serialization() {
        use chrono::Utc;

        let event = RDFEvent::GraphCleared {
            graph: "http://example.com/graph".to_string(),
            timestamp: Utc::now().timestamp_millis(),
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("GraphCleared"));
        assert!(json.contains("http://example.com/graph"));
    }
}
