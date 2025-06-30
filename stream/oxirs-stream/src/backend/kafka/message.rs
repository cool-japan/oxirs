//! Kafka Message Types
//!
//! This module contains message format definitions for the Kafka backend.

use crate::{EventMetadata, StreamEvent};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Enhanced Kafka event with metadata and schema support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaEvent {
    pub schema_id: Option<u32>,
    pub event_id: String,
    pub event_type: String,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub correlation_id: String,
    pub transaction_id: Option<String>,
    pub data: serde_json::Value,
    pub metadata: EventMetadata,
    pub headers: HashMap<String, String>,
    pub partition_key: Option<String>,
    pub schema_version: String,
}

impl From<StreamEvent> for KafkaEvent {
    fn from(event: StreamEvent) -> Self {
        let (event_type, data, metadata, partition_key) = match event {
            StreamEvent::TripleAdded {
                subject,
                predicate,
                object,
                graph,
                metadata,
            } => (
                "triple_added".to_string(),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }),
                metadata,
                Some(subject),
            ),
            StreamEvent::TripleRemoved {
                subject,
                predicate,
                object,
                graph,
                metadata,
            } => (
                "triple_removed".to_string(),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }),
                metadata,
                Some(subject),
            ),
            StreamEvent::QuadAdded {
                subject,
                predicate,
                object,
                graph,
                metadata,
            } => (
                "quad_added".to_string(),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }),
                metadata,
                Some(subject),
            ),
            StreamEvent::QuadRemoved {
                subject,
                predicate,
                object,
                graph,
                metadata,
            } => (
                "quad_removed".to_string(),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }),
                metadata,
                Some(subject),
            ),
            StreamEvent::GraphCreated { graph, metadata } => (
                "graph_created".to_string(),
                serde_json::json!({
                    "graph": graph
                }),
                metadata,
                Some(graph),
            ),
            StreamEvent::GraphCleared { graph, metadata } => (
                "graph_cleared".to_string(),
                serde_json::json!({
                    "graph": graph
                }),
                metadata,
                graph,
            ),
            StreamEvent::GraphDeleted { graph, metadata } => (
                "graph_deleted".to_string(),
                serde_json::json!({
                    "graph": graph
                }),
                metadata,
                Some(graph),
            ),
            StreamEvent::SparqlUpdate {
                query,
                operation_type,
                metadata,
            } => (
                "sparql_update".to_string(),
                serde_json::json!({
                    "query": query,
                    "operation_type": operation_type
                }),
                metadata,
                None,
            ),
            StreamEvent::TransactionBegin {
                transaction_id,
                isolation_level,
                metadata,
            } => (
                "transaction_begin".to_string(),
                serde_json::json!({
                    "transaction_id": transaction_id,
                    "isolation_level": isolation_level
                }),
                metadata,
                Some(transaction_id),
            ),
            StreamEvent::TransactionCommit {
                transaction_id,
                metadata,
            } => (
                "transaction_commit".to_string(),
                serde_json::json!({
                    "transaction_id": transaction_id
                }),
                metadata,
                Some(transaction_id),
            ),
            StreamEvent::TransactionAbort {
                transaction_id,
                metadata,
            } => (
                "transaction_abort".to_string(),
                serde_json::json!({
                    "transaction_id": transaction_id
                }),
                metadata,
                Some(transaction_id),
            ),
            StreamEvent::SchemaChanged {
                schema_type,
                change_type,
                details,
                metadata,
            } => (
                "schema_changed".to_string(),
                serde_json::json!({
                    "schema_type": schema_type,
                    "change_type": change_type,
                    "details": details
                }),
                metadata,
                Some("schema".to_string()),
            ),
            StreamEvent::Heartbeat {
                timestamp,
                source,
                metadata,
            } => (
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
                Some(source),
            ),
            // Catch-all for remaining variants
            _ => (
                "unknown_event".to_string(),
                serde_json::json!({}),
                EventMetadata {
                    event_id: Uuid::new_v4().to_string(),
                    timestamp: Utc::now(),
                    source: "system".to_string(),
                    user: None,
                    context: None,
                    caused_by: None,
                    version: "1.0".to_string(),
                    properties: HashMap::new(),
                    checksum: None,
                },
                None,
            ),
        };

        Self {
            schema_id: None,
            event_id: Uuid::new_v4().to_string(),
            event_type,
            timestamp: Utc::now(),
            source: metadata.source.clone(),
            correlation_id: Uuid::new_v4().to_string(),
            transaction_id: None,
            data,
            metadata,
            headers: HashMap::new(),
            partition_key,
            schema_version: "1.0".to_string(),
        }
    }
}

impl TryFrom<KafkaEvent> for StreamEvent {
    type Error = serde_json::Error;

    fn try_from(kafka_event: KafkaEvent) -> Result<Self, Self::Error> {
        let event = match kafka_event.event_type.as_str() {
            "triple_added" => {
                let subject: String = serde_json::from_value(
                    kafka_event.data.get("subject").unwrap_or(&serde_json::Value::String("".to_string())).clone()
                )?;
                let predicate: String = serde_json::from_value(
                    kafka_event.data.get("predicate").unwrap_or(&serde_json::Value::String("".to_string())).clone()
                )?;
                let object: String = serde_json::from_value(
                    kafka_event.data.get("object").unwrap_or(&serde_json::Value::String("".to_string())).clone()
                )?;
                let graph: Option<String> = serde_json::from_value(
                    kafka_event.data.get("graph").unwrap_or(&serde_json::Value::Null).clone()
                )?;

                StreamEvent::TripleAdded {
                    subject,
                    predicate,
                    object,
                    graph,
                    metadata: kafka_event.metadata,
                }
            }
            "triple_removed" => {
                let subject: String = serde_json::from_value(
                    kafka_event.data.get("subject").unwrap_or(&serde_json::Value::String("".to_string())).clone()
                )?;
                let predicate: String = serde_json::from_value(
                    kafka_event.data.get("predicate").unwrap_or(&serde_json::Value::String("".to_string())).clone()
                )?;
                let object: String = serde_json::from_value(
                    kafka_event.data.get("object").unwrap_or(&serde_json::Value::String("".to_string())).clone()
                )?;
                let graph: Option<String> = serde_json::from_value(
                    kafka_event.data.get("graph").unwrap_or(&serde_json::Value::Null).clone()
                )?;

                StreamEvent::TripleRemoved {
                    subject,
                    predicate,
                    object,
                    graph,
                    metadata: kafka_event.metadata,
                }
            }
            "graph_created" => {
                let graph: String = serde_json::from_value(
                    kafka_event.data.get("graph").unwrap_or(&serde_json::Value::String("".to_string())).clone()
                )?;

                StreamEvent::GraphCreated {
                    graph,
                    metadata: kafka_event.metadata,
                }
            }
            "graph_cleared" => {
                let graph: Option<String> = serde_json::from_value(
                    kafka_event.data.get("graph").unwrap_or(&serde_json::Value::Null).clone()
                )?;

                StreamEvent::GraphCleared {
                    graph,
                    metadata: kafka_event.metadata,
                }
            }
            "graph_deleted" => {
                let graph: String = serde_json::from_value(
                    kafka_event.data.get("graph").unwrap_or(&serde_json::Value::String("".to_string())).clone()
                )?;

                StreamEvent::GraphDeleted {
                    graph,
                    metadata: kafka_event.metadata,
                }
            }
            "transaction_begin" => {
                let transaction_id: String = serde_json::from_value(
                    kafka_event.data.get("transaction_id").unwrap_or(&serde_json::Value::String("".to_string())).clone()
                )?;
                let isolation_level: Option<String> = serde_json::from_value(
                    kafka_event.data.get("isolation_level").unwrap_or(&serde_json::Value::Null).clone()
                )?;

                StreamEvent::TransactionBegin {
                    transaction_id,
                    isolation_level,
                    metadata: kafka_event.metadata,
                }
            }
            "transaction_commit" => {
                let transaction_id: String = serde_json::from_value(
                    kafka_event.data.get("transaction_id").unwrap_or(&serde_json::Value::String("".to_string())).clone()
                )?;

                StreamEvent::TransactionCommit {
                    transaction_id,
                    metadata: kafka_event.metadata,
                }
            }
            "transaction_abort" => {
                let transaction_id: String = serde_json::from_value(
                    kafka_event.data.get("transaction_id").unwrap_or(&serde_json::Value::String("".to_string())).clone()
                )?;

                StreamEvent::TransactionAbort {
                    transaction_id,
                    metadata: kafka_event.metadata,
                }
            }
            "heartbeat" => {
                let source: String = serde_json::from_value(
                    kafka_event.data.get("source").unwrap_or(&serde_json::Value::String("".to_string())).clone()
                )?;

                StreamEvent::Heartbeat {
                    timestamp: kafka_event.timestamp,
                    source,
                    metadata: kafka_event.metadata,
                }
            }
            _ => {
                // For unknown event types, create a generic event
                StreamEvent::Heartbeat {
                    timestamp: kafka_event.timestamp,
                    source: kafka_event.source,
                    metadata: kafka_event.metadata,
                }
            }
        };

        Ok(event)
    }
}

impl KafkaEvent {
    /// Create a new Kafka event
    pub fn new(event_type: String, data: serde_json::Value, metadata: EventMetadata) -> Self {
        Self {
            schema_id: None,
            event_id: Uuid::new_v4().to_string(),
            event_type,
            timestamp: Utc::now(),
            source: metadata.source.clone(),
            correlation_id: Uuid::new_v4().to_string(),
            transaction_id: None,
            data,
            metadata,
            headers: HashMap::new(),
            partition_key: None,
            schema_version: "1.0".to_string(),
        }
    }

    /// Set schema information
    pub fn with_schema(mut self, schema_id: u32, schema_version: String) -> Self {
        self.schema_id = Some(schema_id);
        self.schema_version = schema_version;
        self
    }

    /// Set partition key for message routing
    pub fn with_partition_key(mut self, key: String) -> Self {
        self.partition_key = Some(key);
        self
    }

    /// Add header to the message
    pub fn with_header(mut self, key: String, value: String) -> Self {
        self.headers.insert(key, value);
        self
    }

    /// Set transaction ID for transactional messaging
    pub fn with_transaction_id(mut self, transaction_id: String) -> Self {
        self.transaction_id = Some(transaction_id);
        self
    }

    /// Convert to bytes for sending
    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }

    /// Validate event format
    pub fn validate(&self) -> Result<(), String> {
        if self.event_id.is_empty() {
            return Err("Event ID cannot be empty".to_string());
        }
        if self.event_type.is_empty() {
            return Err("Event type cannot be empty".to_string());
        }
        if self.source.is_empty() {
            return Err("Source cannot be empty".to_string());
        }
        Ok(())
    }

    /// Get message size in bytes
    pub fn size(&self) -> usize {
        self.to_bytes().map(|b| b.len()).unwrap_or(0)
    }

    /// Get topic name based on event type
    pub fn get_topic_name(&self, prefix: &str) -> String {
        format!("{}.{}", prefix, self.event_type)
    }
}