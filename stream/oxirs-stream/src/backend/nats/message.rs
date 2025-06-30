//! NATS Message Types
//!
//! This module contains message format definitions for the NATS backend.

use crate::{EventMetadata, PatchOperation, RdfPatch, StreamEvent};
use serde::{Deserialize, Serialize};

/// NATS event message format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NatsEventMessage {
    pub event_id: String,
    pub event_type: String,
    pub timestamp: u64,
    pub data: serde_json::Value,
    pub metadata: Option<EventMetadata>,
}

impl From<StreamEvent> for NatsEventMessage {
    fn from(event: StreamEvent) -> Self {
        let event_type = match &event {
            StreamEvent::TripleAdded { .. } => "triple_added",
            StreamEvent::TripleRemoved { .. } => "triple_removed",
            StreamEvent::GraphCleared { .. } => "graph_cleared",
            StreamEvent::PatchApplied { .. } => "patch_applied",
            StreamEvent::TransactionStarted { .. } => "transaction_started",
            StreamEvent::TransactionCommitted { .. } => "transaction_committed",
            StreamEvent::TransactionAborted { .. } => "transaction_aborted",
        };

        let data = serde_json::to_value(&event).unwrap_or_default();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            event_id: uuid::Uuid::new_v4().to_string(),
            event_type: event_type.to_string(),
            timestamp,
            data,
            metadata: event.metadata().cloned(),
        }
    }
}

impl NatsEventMessage {
    /// Create a new NATS event message
    pub fn new(event_type: String, data: serde_json::Value) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            event_id: uuid::Uuid::new_v4().to_string(),
            event_type,
            timestamp,
            data,
            metadata: None,
        }
    }

    /// Set metadata for the message
    pub fn with_metadata(mut self, metadata: EventMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Get the message subject based on event type
    pub fn get_subject(&self, prefix: &str) -> String {
        format!("{}.{}", prefix, self.event_type)
    }

    /// Convert to bytes for sending
    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }

    /// Validate message format
    pub fn validate(&self) -> Result<(), String> {
        if self.event_id.is_empty() {
            return Err("Event ID cannot be empty".to_string());
        }
        if self.event_type.is_empty() {
            return Err("Event type cannot be empty".to_string());
        }
        if self.timestamp == 0 {
            return Err("Timestamp cannot be zero".to_string());
        }
        Ok(())
    }

    /// Get message size in bytes
    pub fn size(&self) -> usize {
        self.to_bytes().map(|b| b.len()).unwrap_or(0)
    }
}

/// Conversion from NatsEventMessage to StreamEvent
impl TryFrom<NatsEventMessage> for StreamEvent {
    type Error = serde_json::Error;

    fn try_from(msg: NatsEventMessage) -> Result<Self, Self::Error> {
        serde_json::from_value(msg.data)
    }
}