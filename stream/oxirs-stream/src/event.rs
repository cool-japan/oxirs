//! # Stream Events
//!
//! Event types for RDF streaming.

use serde::{Deserialize, Serialize};
use crate::types::EventMetadata;

/// RDF streaming event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEvent {
    pub event_type: StreamEventType,
    pub timestamp: u64,
    pub metadata: Option<EventMetadata>,
}

/// Event type variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamEventType {
    TripleAdded {
        subject: String,
        predicate: String,
        object: String,
        graph: Option<String>,
    },
    TripleRemoved {
        subject: String,
        predicate: String,
        object: String,
        graph: Option<String>,
    },
    QuadAdded {
        subject: String,
        predicate: String,
        object: String,
        graph: String,
    },
    QuadRemoved {
        subject: String,
        predicate: String,
        object: String,
        graph: String,
    },
    GraphCreated {
        graph: String,
    },
    GraphCleared {
        graph: Option<String>,
    },
    GraphDeleted {
        graph: String,
    },
    SparqlUpdate {
        query: String,
    },
    TransactionBegin {
        transaction_id: String,
    },
    TransactionCommit {
        transaction_id: String,
    },
    TransactionAbort {
        transaction_id: String,
    },
}

impl StreamEvent {
    /// Create a new stream event
    pub fn new(event_type: StreamEventType) -> Self {
        Self {
            event_type,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            metadata: None,
        }
    }

    /// Create an event with metadata
    pub fn with_metadata(event_type: StreamEventType, metadata: EventMetadata) -> Self {
        Self {
            event_type,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            metadata: Some(metadata),
        }
    }
}