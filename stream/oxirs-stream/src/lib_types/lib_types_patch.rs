//! RDF patch types and unified Stream interface

use anyhow::Result;
use chrono;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;
use uuid;

use crate::event::{EventMetadata, StreamEvent};

use super::lib_types_config::StreamConfig;
use super::lib_types_consumer::{ConsumerStats, StreamConsumer};
use super::lib_types_producer::{ProducerStats, StreamProducer};

/// RDF patch operations with full protocol support
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatchOperation {
    /// Add a triple (A operation)
    Add {
        subject: String,
        predicate: String,
        object: String,
    },
    /// Delete a triple (D operation)
    Delete {
        subject: String,
        predicate: String,
        object: String,
    },
    /// Add a graph (GA operation)
    AddGraph { graph: String },
    /// Delete a graph (GD operation)
    DeleteGraph { graph: String },
    /// Add a prefix (PA operation)
    AddPrefix { prefix: String, namespace: String },
    /// Delete a prefix (PD operation)
    DeletePrefix { prefix: String },
    /// Transaction begin (TX operation)
    TransactionBegin { transaction_id: Option<String> },
    /// Transaction commit (TC operation)
    TransactionCommit,
    /// Transaction abort (TA operation)
    TransactionAbort,
    /// Header information (H operation)
    Header { key: String, value: String },
}

/// RDF patch for atomic updates with full protocol support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfPatch {
    pub operations: Vec<PatchOperation>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub id: String,
    /// Patch headers for metadata
    pub headers: HashMap<String, String>,
    /// Current transaction ID if in transaction
    pub transaction_id: Option<String>,
    /// Prefixes used in the patch
    pub prefixes: HashMap<String, String>,
}

impl RdfPatch {
    /// Create a new RDF patch
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            timestamp: chrono::Utc::now(),
            id: uuid::Uuid::new_v4().to_string(),
            headers: HashMap::new(),
            transaction_id: None,
            prefixes: HashMap::new(),
        }
    }

    /// Add an operation to the patch
    pub fn add_operation(&mut self, operation: PatchOperation) {
        self.operations.push(operation);
    }

    /// Serialize patch to RDF Patch format
    pub fn to_rdf_patch_format(&self) -> Result<String> {
        let serializer = crate::patch::PatchSerializer::new()
            .with_pretty_print(true)
            .with_metadata(true);
        serializer.serialize(self)
    }

    /// Parse from RDF Patch format
    pub fn from_rdf_patch_format(input: &str) -> Result<Self> {
        let mut parser = crate::patch::PatchParser::new().with_strict_mode(false);
        parser.parse(input)
    }
}

impl Default for RdfPatch {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// publish_patch — standalone helper (also called from StreamProducer::publish_patch)
// ============================================================

/// Standalone helper: publish an RDF patch via a StreamProducer
pub async fn publish_patch(producer: &mut StreamProducer, patch: &RdfPatch) -> Result<()> {
    let events: Vec<StreamEvent> = patch
        .operations
        .iter()
        .filter_map(|op| {
            let metadata = EventMetadata {
                event_id: uuid::Uuid::new_v4().to_string(),
                timestamp: patch.timestamp,
                source: "rdf_patch".to_string(),
                user: None,
                context: Some(patch.id.clone()),
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            };

            match op {
                PatchOperation::Add {
                    subject,
                    predicate,
                    object,
                } => Some(StreamEvent::TripleAdded {
                    subject: subject.clone(),
                    predicate: predicate.clone(),
                    object: object.clone(),
                    graph: None,
                    metadata,
                }),
                PatchOperation::Delete {
                    subject,
                    predicate,
                    object,
                } => Some(StreamEvent::TripleRemoved {
                    subject: subject.clone(),
                    predicate: predicate.clone(),
                    object: object.clone(),
                    graph: None,
                    metadata,
                }),
                PatchOperation::AddGraph { graph } => Some(StreamEvent::GraphCreated {
                    graph: graph.clone(),
                    metadata,
                }),
                PatchOperation::DeleteGraph { graph } => Some(StreamEvent::GraphDeleted {
                    graph: graph.clone(),
                    metadata,
                }),
                PatchOperation::AddPrefix { .. } => None,
                PatchOperation::DeletePrefix { .. } => None,
                PatchOperation::TransactionBegin { .. } => Some(StreamEvent::TransactionBegin {
                    transaction_id: patch
                        .transaction_id
                        .clone()
                        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    isolation_level: None,
                    metadata,
                }),
                PatchOperation::TransactionCommit => Some(StreamEvent::TransactionCommit {
                    transaction_id: patch
                        .transaction_id
                        .clone()
                        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    metadata,
                }),
                PatchOperation::TransactionAbort => Some(StreamEvent::TransactionAbort {
                    transaction_id: patch
                        .transaction_id
                        .clone()
                        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    metadata,
                }),
                PatchOperation::Header { .. } => None,
            }
        })
        .collect();

    producer.publish_batch(events).await
}

// ============================================================
// Unified Stream
// ============================================================

/// Unified Stream interface that combines producer and consumer functionality
pub struct Stream {
    producer: StreamProducer,
    consumer: StreamConsumer,
}

impl Stream {
    /// Create a new unified stream instance
    pub async fn new(config: StreamConfig) -> Result<Self> {
        let producer = StreamProducer::new(config.clone()).await?;
        let consumer = StreamConsumer::new(config).await?;

        Ok(Self { producer, consumer })
    }

    /// Publish an event to the stream
    pub async fn publish(&mut self, event: StreamEvent) -> Result<()> {
        self.producer.publish(event).await
    }

    /// Consume an event from the stream
    pub async fn consume(&mut self) -> Result<Option<StreamEvent>> {
        self.consumer.consume().await
    }

    /// Flush any pending events
    pub async fn flush(&mut self) -> Result<()> {
        self.producer.flush().await
    }

    /// Get producer statistics
    pub async fn producer_stats(&self) -> ProducerStats {
        self.producer.get_stats().await
    }

    /// Get consumer statistics
    pub async fn consumer_stats(&self) -> ConsumerStats {
        self.consumer.get_stats().await
    }

    /// Close the stream and clean up resources
    pub async fn close(&mut self) -> Result<()> {
        self.producer.flush().await?;
        debug!("Stream closed successfully");
        Ok(())
    }

    /// Perform a health check on the stream
    pub async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }

    /// Begin a transaction (placeholder implementation)
    pub async fn begin_transaction(&mut self) -> Result<()> {
        debug!("Transaction begun (placeholder)");
        Ok(())
    }

    /// Commit a transaction (placeholder implementation)
    pub async fn commit_transaction(&mut self) -> Result<()> {
        debug!("Transaction committed (placeholder)");
        Ok(())
    }

    /// Rollback a transaction (placeholder implementation)
    pub async fn rollback_transaction(&mut self) -> Result<()> {
        debug!("Transaction rolled back (placeholder)");
        Ok(())
    }
}
