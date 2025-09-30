//! NATS Message Types
//!
//! This module contains NATS-specific message types and conversion implementations.

use crate::{EventMetadata, StreamEvent};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// NATS event message format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NatsEventMessage {
    pub event_id: String,
    pub event_type: String,
    pub timestamp: u64,
    pub data: serde_json::Value,
    pub metadata: Option<EventMetadata>,
}

/// Consumer state for tracking offsets and acknowledgments
#[derive(Debug, Clone)]
pub struct ConsumerState {
    pub sequence: u64,
    pub timestamp: u64,
    pub pending_acks: HashMap<String, u64>,
    pub last_activity: std::time::Instant,
    pub is_healthy: bool,
}

/// Circuit breaker state for reliability
#[derive(Debug, Clone)]
pub struct CircuitBreakerState {
    pub failures: u32,
    pub state: CircuitState,
    pub last_failure: Option<std::time::Instant>,
    pub last_success: Option<std::time::Instant>,
}

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Connection pool metrics
#[derive(Debug, Clone)]
pub struct ConnectionMetrics {
    pub active_connections: usize,
    pub total_connections: usize,
    pub failed_connections: usize,
    pub last_connection_time: Option<std::time::Instant>,
}

impl From<StreamEvent> for NatsEventMessage {
    fn from(event: StreamEvent) -> Self {
        let (event_type, data, metadata) = match event {
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
                Some(metadata),
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
                Some(metadata),
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
                Some(metadata),
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
                Some(metadata),
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
                Some(metadata),
            ),
            StreamEvent::TransactionCommit {
                transaction_id,
                metadata,
            } => (
                "transaction_commit".to_string(),
                serde_json::json!({
                    "transaction_id": transaction_id
                }),
                Some(metadata),
            ),
            StreamEvent::TransactionAbort {
                transaction_id,
                metadata,
            } => (
                "transaction_abort".to_string(),
                serde_json::json!({
                    "transaction_id": transaction_id
                }),
                Some(metadata),
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
                Some(metadata),
            ),
            StreamEvent::Heartbeat {
                timestamp,
                source,
                metadata,
            } => (
                "heartbeat".to_string(),
                serde_json::json!({
                    "source": source,
                    "timestamp": timestamp
                }),
                None,
            ),
            // Catch-all for remaining variants
            _ => ("unknown_event".to_string(), serde_json::json!({}), None),
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
    pub fn to_stream_event(&self) -> Result<StreamEvent> {
        let metadata = self.metadata.clone().unwrap_or_default();

        let event = match self.event_type.as_str() {
            "triple_added" => {
                let subject = self.data["subject"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing subject"))?
                    .to_string();
                let predicate = self.data["predicate"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing predicate"))?
                    .to_string();
                let object = self.data["object"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing object"))?
                    .to_string();
                let graph = self.data["graph"].as_str().map(|s| s.to_string());

                StreamEvent::TripleAdded {
                    subject,
                    predicate,
                    object,
                    graph,
                    metadata,
                }
            }
            "triple_removed" => {
                let subject = self.data["subject"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing subject"))?
                    .to_string();
                let predicate = self.data["predicate"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing predicate"))?
                    .to_string();
                let object = self.data["object"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing object"))?
                    .to_string();
                let graph = self.data["graph"].as_str().map(|s| s.to_string());

                StreamEvent::TripleRemoved {
                    subject,
                    predicate,
                    object,
                    graph,
                    metadata,
                }
            }
            "heartbeat" => {
                let source = self.data["source"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing source"))?
                    .to_string();
                let timestamp_u64 = self.data["timestamp"]
                    .as_u64()
                    .ok_or_else(|| anyhow!("Missing timestamp"))?;
                let timestamp = chrono::DateTime::from_timestamp(timestamp_u64 as i64, 0)
                    .ok_or_else(|| anyhow!("Invalid timestamp"))?;

                StreamEvent::Heartbeat {
                    timestamp,
                    source,
                    metadata,
                }
            }
            _ => {
                return Err(anyhow!("Unknown event type: {}", self.event_type));
            }
        };

        Ok(event)
    }
}

impl Default for ConsumerState {
    fn default() -> Self {
        Self {
            sequence: 0,
            timestamp: 0,
            pending_acks: HashMap::new(),
            last_activity: std::time::Instant::now(),
            is_healthy: true,
        }
    }
}

impl Default for CircuitBreakerState {
    fn default() -> Self {
        Self {
            failures: 0,
            state: CircuitState::Closed,
            last_failure: None,
            last_success: None,
        }
    }
}
