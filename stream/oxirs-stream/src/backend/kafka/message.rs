//! Kafka message types and conversion utilities

use crate::{EventMetadata, StreamEvent};
use anyhow::{anyhow, Result};
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

impl KafkaEvent {
    /// Convert KafkaEvent to StreamEvent
    pub fn to_stream_event(self) -> StreamEvent {
        match self.event_type.as_str() {
            "triple_added" => StreamEvent::TripleAdded {
                subject: self.data["subject"].as_str().unwrap_or("").to_string(),
                predicate: self.data["predicate"].as_str().unwrap_or("").to_string(),
                object: self.data["object"].as_str().unwrap_or("").to_string(),
                graph: self.data["graph"].as_str().map(|s| s.to_string()),
                metadata: self.metadata,
            },
            "triple_removed" => StreamEvent::TripleRemoved {
                subject: self.data["subject"].as_str().unwrap_or("").to_string(),
                predicate: self.data["predicate"].as_str().unwrap_or("").to_string(),
                object: self.data["object"].as_str().unwrap_or("").to_string(),
                graph: self.data["graph"].as_str().map(|s| s.to_string()),
                metadata: self.metadata,
            },
            "graph_created" => StreamEvent::GraphCreated {
                graph_uri: self.data["graph_uri"].as_str().unwrap_or("").to_string(),
                metadata: self.metadata,
            },
            "sparql_update" => StreamEvent::SparqlUpdate {
                query: self.data["query"].as_str().unwrap_or("").to_string(),
                metadata: self.metadata,
            },
            "transaction_begin" => StreamEvent::TransactionBegin {
                transaction_id: self.data["transaction_id"].as_str().unwrap_or("").to_string(),
                metadata: self.metadata,
            },
            "transaction_commit" => StreamEvent::TransactionCommit {
                transaction_id: self.data["transaction_id"].as_str().unwrap_or("").to_string(),
                metadata: self.metadata,
            },
            "transaction_abort" => StreamEvent::TransactionAbort {
                transaction_id: self.data["transaction_id"].as_str().unwrap_or("").to_string(),
                metadata: self.metadata,
            },
            "heartbeat" => StreamEvent::Heartbeat {
                timestamp: self.timestamp,
                source: self.data["source"].as_str().unwrap_or("").to_string(),
                metadata: self.metadata,
            },
            _ => StreamEvent::Heartbeat {
                timestamp: self.timestamp,
                source: "unknown".to_string(),
                metadata: self.metadata,
            },
        }
    }

    /// Convert from StreamEvent to KafkaEvent
    pub fn from_stream_event(event: StreamEvent) -> Self {
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
            StreamEvent::GraphCreated {
                graph_uri,
                metadata,
            } => (
                "graph_created".to_string(),
                serde_json::json!({
                    "graph_uri": graph_uri
                }),
                metadata,
                Some(graph_uri),
            ),
            StreamEvent::SparqlUpdate { query, metadata } => (
                "sparql_update".to_string(),
                serde_json::json!({
                    "query": query
                }),
                metadata,
                None,
            ),
            StreamEvent::TransactionBegin {
                transaction_id,
                metadata,
            } => (
                "transaction_begin".to_string(),
                serde_json::json!({
                    "transaction_id": transaction_id
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
            StreamEvent::Heartbeat {
                timestamp,
                source,
                metadata,
            } => (
                "heartbeat".to_string(),
                serde_json::json!({
                    "source": source
                }),
                metadata,
                Some(source),
            ),
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
            event_id: metadata.event_id.clone(),
            event_type,
            timestamp: metadata.timestamp,
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

    /// Get topic name for this event
    pub fn get_topic_name(&self, base_topic: &str) -> String {
        format!("{}_{}", base_topic, self.event_type)
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| anyhow!("Failed to serialize KafkaEvent: {}", e))
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(|e| anyhow!("Failed to deserialize KafkaEvent: {}", e))
    }
}

impl From<StreamEvent> for KafkaEvent {
    fn from(event: StreamEvent) -> Self {
        Self::from_stream_event(event)
    }
}

impl TryFrom<KafkaEvent> for StreamEvent {
    type Error = anyhow::Error;

    fn try_from(kafka_event: KafkaEvent) -> Result<Self> {
        Ok(kafka_event.to_stream_event())
    }
}