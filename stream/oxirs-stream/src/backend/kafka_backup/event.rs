//! Kafka event types and conversions

use crate::event::StreamEvent;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Kafka-specific event wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaEvent {
    pub id: String,
    pub timestamp: i64,
    pub event_type: String,
    pub source: String,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub graph: Option<String>,
    pub operation: String,
    pub metadata: HashMap<String, String>,
    pub schema_version: String,
    pub content_type: String,
}

impl From<StreamEvent> for KafkaEvent {
    fn from(event: StreamEvent) -> Self {
        match event {
            StreamEvent::TripleAdded { subject, predicate, object, graph, metadata } => {
                KafkaEvent {
                    id: uuid::Uuid::new_v4().to_string(),
                    timestamp: chrono::Utc::now().timestamp_millis(),
                    event_type: "TripleAdded".to_string(),
                    source: metadata.source.unwrap_or_else(|| "unknown".to_string()),
                    subject: Some(subject),
                    predicate: Some(predicate),
                    object: Some(object),
                    graph,
                    operation: "ADD".to_string(),
                    metadata: metadata.custom,
                    schema_version: "1.0".to_string(),
                    content_type: "application/rdf+json".to_string(),
                }
            }
            StreamEvent::TripleRemoved { subject, predicate, object, graph, metadata } => {
                KafkaEvent {
                    id: uuid::Uuid::new_v4().to_string(),
                    timestamp: chrono::Utc::now().timestamp_millis(),
                    event_type: "TripleRemoved".to_string(),
                    source: metadata.source.unwrap_or_else(|| "unknown".to_string()),
                    subject: Some(subject),
                    predicate: Some(predicate),
                    object: Some(object),
                    graph,
                    operation: "REMOVE".to_string(),
                    metadata: metadata.custom,
                    schema_version: "1.0".to_string(),
                    content_type: "application/rdf+json".to_string(),
                }
            }
            _ => {
                KafkaEvent {
                    id: uuid::Uuid::new_v4().to_string(),
                    timestamp: chrono::Utc::now().timestamp_millis(),
                    event_type: "Unknown".to_string(),
                    source: "unknown".to_string(),
                    subject: None,
                    predicate: None,
                    object: None,
                    graph: None,
                    operation: "UNKNOWN".to_string(),
                    metadata: HashMap::new(),
                    schema_version: "1.0".to_string(),
                    content_type: "application/json".to_string(),
                }
            }
        }
    }
}

impl TryFrom<KafkaEvent> for StreamEvent {
    type Error = String;

    fn try_from(kafka_event: KafkaEvent) -> Result<Self, Self::Error> {
        match kafka_event.event_type.as_str() {
            "TripleAdded" => {
                let subject = kafka_event.subject.ok_or("Missing subject")?;
                let predicate = kafka_event.predicate.ok_or("Missing predicate")?;
                let object = kafka_event.object.ok_or("Missing object")?;
                
                Ok(StreamEvent::TripleAdded {
                    subject,
                    predicate,
                    object,
                    graph: kafka_event.graph,
                    metadata: crate::EventMetadata {
                        timestamp: chrono::DateTime::from_timestamp_millis(kafka_event.timestamp)
                            .unwrap_or_else(chrono::Utc::now),
                        source: Some(kafka_event.source),
                        custom: kafka_event.metadata,
                        ..Default::default()
                    },
                })
            }
            "TripleRemoved" => {
                let subject = kafka_event.subject.ok_or("Missing subject")?;
                let predicate = kafka_event.predicate.ok_or("Missing predicate")?;
                let object = kafka_event.object.ok_or("Missing object")?;
                
                Ok(StreamEvent::TripleRemoved {
                    subject,
                    predicate,
                    object,
                    graph: kafka_event.graph,
                    metadata: crate::EventMetadata {
                        timestamp: chrono::DateTime::from_timestamp_millis(kafka_event.timestamp)
                            .unwrap_or_else(chrono::Utc::now),
                        source: Some(kafka_event.source),
                        custom: kafka_event.metadata,
                        ..Default::default()
                    },
                })
            }
            _ => Err(format!("Unknown event type: {}", kafka_event.event_type))
        }
    }
}