//! Kafka message types and serialization

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
    /// Get topic name for this event
    pub fn get_topic_name(&self, prefix: &str) -> String {
        format!("{}-{}", prefix, self.event_type.replace('_', "-"))
    }

    /// Convert to bytes for Kafka payload
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| anyhow!("Failed to serialize KafkaEvent: {}", e))
    }

    /// Parse from bytes (Kafka payload deserialization)
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes)
            .map_err(|e| anyhow!("Failed to deserialize KafkaEvent: {}", e))
    }

    /// Create from StreamEvent for publishing
    pub fn from_stream_event(event: StreamEvent) -> Self {
        event.into()
    }

    /// Convert to StreamEvent for consumption
    pub fn to_stream_event(self) -> StreamEvent {
        let timestamp = self.timestamp;
        let source = self.source.clone();
        let metadata = self.metadata.clone();

        self.try_into().unwrap_or({
            // Fallback to a default event if conversion fails
            StreamEvent::Heartbeat {
                timestamp,
                source,
                metadata,
            }
        })
    }
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
                metadata: _,
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
}

impl TryFrom<KafkaEvent> for StreamEvent {
    type Error = anyhow::Error;

    fn try_from(kafka_event: KafkaEvent) -> Result<Self> {
        let metadata = kafka_event.metadata;

        match kafka_event.event_type.as_str() {
            "triple_added" => {
                let subject = kafka_event.data["subject"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing subject"))?
                    .to_string();
                let predicate = kafka_event.data["predicate"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing predicate"))?
                    .to_string();
                let object = kafka_event.data["object"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing object"))?
                    .to_string();
                let graph = kafka_event.data["graph"].as_str().map(|s| s.to_string());

                Ok(StreamEvent::TripleAdded {
                    subject,
                    predicate,
                    object,
                    graph,
                    metadata,
                })
            }
            "triple_removed" => {
                let subject = kafka_event.data["subject"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing subject"))?
                    .to_string();
                let predicate = kafka_event.data["predicate"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing predicate"))?
                    .to_string();
                let object = kafka_event.data["object"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing object"))?
                    .to_string();
                let graph = kafka_event.data["graph"].as_str().map(|s| s.to_string());

                Ok(StreamEvent::TripleRemoved {
                    subject,
                    predicate,
                    object,
                    graph,
                    metadata,
                })
            }
            "quad_added" => {
                let subject = kafka_event.data["subject"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing subject"))?
                    .to_string();
                let predicate = kafka_event.data["predicate"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing predicate"))?
                    .to_string();
                let object = kafka_event.data["object"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing object"))?
                    .to_string();
                let graph = kafka_event.data["graph"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing graph"))?
                    .to_string();

                Ok(StreamEvent::QuadAdded {
                    subject,
                    predicate,
                    object,
                    graph,
                    metadata,
                })
            }
            "quad_removed" => {
                let subject = kafka_event.data["subject"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing subject"))?
                    .to_string();
                let predicate = kafka_event.data["predicate"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing predicate"))?
                    .to_string();
                let object = kafka_event.data["object"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing object"))?
                    .to_string();
                let graph = kafka_event.data["graph"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing graph"))?
                    .to_string();

                Ok(StreamEvent::QuadRemoved {
                    subject,
                    predicate,
                    object,
                    graph,
                    metadata,
                })
            }
            "graph_created" => {
                let graph = kafka_event.data["graph"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing graph"))?
                    .to_string();

                Ok(StreamEvent::GraphCreated { graph, metadata })
            }
            "graph_cleared" => {
                let graph = kafka_event.data["graph"].as_str().map(|s| s.to_string());

                Ok(StreamEvent::GraphCleared { graph, metadata })
            }
            "graph_deleted" => {
                let graph = kafka_event.data["graph"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing graph"))?
                    .to_string();

                Ok(StreamEvent::GraphDeleted { graph, metadata })
            }
            "heartbeat" => {
                let source = kafka_event.data["source"]
                    .as_str()
                    .unwrap_or(&kafka_event.source)
                    .to_string();

                Ok(StreamEvent::Heartbeat {
                    timestamp: kafka_event.timestamp,
                    source,
                    metadata,
                })
            }
            _ => Err(anyhow!("Unknown event type: {}", kafka_event.event_type)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kafka_event_serialization() {
        let event = StreamEvent::TripleAdded {
            subject: "test:subject".to_string(),
            predicate: "test:predicate".to_string(),
            object: "test:object".to_string(),
            graph: Some("test:graph".to_string()),
            metadata: EventMetadata {
                event_id: "test-id".to_string(),
                timestamp: Utc::now(),
                source: "test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            },
        };

        let kafka_event = KafkaEvent::from(event);
        assert_eq!(kafka_event.event_type, "triple_added");

        let bytes = kafka_event.to_bytes().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_kafka_event_roundtrip() {
        let original_event = StreamEvent::TripleAdded {
            subject: "test:subject".to_string(),
            predicate: "test:predicate".to_string(),
            object: "test:object".to_string(),
            graph: Some("test:graph".to_string()),
            metadata: EventMetadata {
                event_id: "test-id".to_string(),
                timestamp: Utc::now(),
                source: "test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            },
        };

        let kafka_event = KafkaEvent::from(original_event);
        let roundtrip_event: StreamEvent = kafka_event.try_into().unwrap();

        match roundtrip_event {
            StreamEvent::TripleAdded {
                subject,
                predicate,
                object,
                graph,
                ..
            } => {
                assert_eq!(subject, "test:subject");
                assert_eq!(predicate, "test:predicate");
                assert_eq!(object, "test:object");
                assert_eq!(graph, Some("test:graph".to_string()));
            }
            _ => panic!("Unexpected event type after roundtrip"),
        }
    }
}
