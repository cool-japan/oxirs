//! # Stream Events
//!
//! Event types for RDF streaming with comprehensive metadata and provenance tracking.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use uuid;

// Event metadata is defined as a struct below

/// Enhanced RDF streaming events with metadata and provenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamEvent {
    TripleAdded {
        subject: String,
        predicate: String,
        object: String,
        graph: Option<String>,
        metadata: EventMetadata,
    },
    TripleRemoved {
        subject: String,
        predicate: String,
        object: String,
        graph: Option<String>,
        metadata: EventMetadata,
    },
    QuadAdded {
        subject: String,
        predicate: String,
        object: String,
        graph: String,
        metadata: EventMetadata,
    },
    QuadRemoved {
        subject: String,
        predicate: String,
        object: String,
        graph: String,
        metadata: EventMetadata,
    },
    GraphCreated {
        graph: String,
        metadata: EventMetadata,
    },
    GraphCleared {
        graph: Option<String>,
        metadata: EventMetadata,
    },
    GraphDeleted {
        graph: String,
        metadata: EventMetadata,
    },
    SparqlUpdate {
        query: String,
        operation_type: SparqlOperationType,
        metadata: EventMetadata,
    },
    TransactionBegin {
        transaction_id: String,
        isolation_level: Option<IsolationLevel>,
        metadata: EventMetadata,
    },
    TransactionCommit {
        transaction_id: String,
        metadata: EventMetadata,
    },
    TransactionAbort {
        transaction_id: String,
        metadata: EventMetadata,
    },
    SchemaChanged {
        schema_type: SchemaType,
        change_type: SchemaChangeType,
        details: String,
        metadata: EventMetadata,
    },
    Heartbeat {
        timestamp: DateTime<Utc>,
        source: String,
        metadata: EventMetadata,
    },
    QueryResultAdded {
        query_id: String,
        result: QueryResult,
        metadata: EventMetadata,
    },
    QueryResultRemoved {
        query_id: String,
        result: QueryResult,
        metadata: EventMetadata,
    },
    QueryCompleted {
        query_id: String,
        execution_time: Duration,
        metadata: EventMetadata,
    },

    // Named Graph Events
    GraphMetadataUpdated {
        graph: String,
        metadata_type: String,
        old_value: Option<String>,
        new_value: String,
        metadata: EventMetadata,
    },
    GraphPermissionsChanged {
        graph: String,
        permission_type: String, // "read", "write", "admin"
        principal: String,       // user or role
        granted: bool,
        metadata: EventMetadata,
    },
    GraphStatisticsUpdated {
        graph: String,
        triple_count: u64,
        size_bytes: u64,
        last_modified: u64,
        metadata: EventMetadata,
    },
    GraphRenamed {
        old_name: String,
        new_name: String,
        metadata: EventMetadata,
    },
    GraphMerged {
        source_graphs: Vec<String>,
        target_graph: String,
        metadata: EventMetadata,
    },
    GraphSplit {
        source_graph: String,
        target_graphs: Vec<String>,
        split_criteria: String,
        metadata: EventMetadata,
    },

    // Schema Change Events
    SchemaDefinitionAdded {
        schema_type: String, // "class", "property", "datatype"
        schema_uri: String,
        definition: String,
        metadata: EventMetadata,
    },
    SchemaDefinitionRemoved {
        schema_type: String,
        schema_uri: String,
        metadata: EventMetadata,
    },
    SchemaDefinitionModified {
        schema_type: String,
        schema_uri: String,
        old_definition: String,
        new_definition: String,
        metadata: EventMetadata,
    },
    OntologyImported {
        ontology_uri: String,
        version: Option<String>,
        import_method: String, // "owl:imports", "explicit", "inference"
        metadata: EventMetadata,
    },
    OntologyRemoved {
        ontology_uri: String,
        version: Option<String>,
        metadata: EventMetadata,
    },
    ConstraintAdded {
        constraint_type: String, // "cardinality", "range", "domain", "functional"
        target: String,          // property or class URI
        constraint_definition: String,
        metadata: EventMetadata,
    },
    ConstraintRemoved {
        constraint_type: String,
        target: String,
        constraint_definition: String,
        metadata: EventMetadata,
    },
    ConstraintViolated {
        constraint_type: String,
        target: String,
        violating_data: String,
        severity: String, // "error", "warning", "info"
        metadata: EventMetadata,
    },
    IndexCreated {
        index_name: String,
        index_type: String, // "btree", "hash", "fulltext", "spatial"
        target_properties: Vec<String>,
        graph: Option<String>,
        metadata: EventMetadata,
    },
    IndexDropped {
        index_name: String,
        index_type: String,
        metadata: EventMetadata,
    },
    IndexRebuilt {
        index_name: String,
        reason: String,
        duration_ms: u64,
        metadata: EventMetadata,
    },

    // Schema Update Events
    SchemaUpdated {
        schema_uri: String,
        update_type: String,
        old_definition: Option<String>,
        new_definition: String,
        metadata: EventMetadata,
    },

    // SHACL Shape Events
    ShapeAdded {
        shape_uri: String,
        shape_definition: String,
        target_class: Option<String>,
        metadata: EventMetadata,
    },
    ShapeUpdated {
        shape_uri: String,
        old_definition: String,
        new_definition: String,
        target_class: Option<String>,
        metadata: EventMetadata,
    },
    ShapeRemoved {
        shape_uri: String,
        metadata: EventMetadata,
    },
    ShapeModified {
        shape_uri: String,
        old_definition: String,
        new_definition: String,
        metadata: EventMetadata,
    },
    ShapeValidationStarted {
        shape_uri: String,
        target_graph: Option<String>,
        validation_id: String,
        metadata: EventMetadata,
    },
    ShapeValidationCompleted {
        shape_uri: String,
        validation_id: String,
        success: bool,
        violation_count: u32,
        duration_ms: u64,
        metadata: EventMetadata,
    },
    ShapeViolationDetected {
        shape_uri: String,
        violation_path: String,
        violating_node: String,
        severity: String,
        message: String,
        metadata: EventMetadata,
    },
    
    // Error Events
    ErrorOccurred {
        error_type: String,
        error_message: String,
        error_context: Option<String>,
        metadata: EventMetadata,
    },
}

impl StreamEvent {
    /// Extract timestamp from any StreamEvent variant
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            StreamEvent::TripleAdded { metadata, .. } => metadata.timestamp,
            StreamEvent::TripleRemoved { metadata, .. } => metadata.timestamp,
            StreamEvent::QuadAdded { metadata, .. } => metadata.timestamp,
            StreamEvent::QuadRemoved { metadata, .. } => metadata.timestamp,
            StreamEvent::GraphCreated { metadata, .. } => metadata.timestamp,
            StreamEvent::GraphCleared { metadata, .. } => metadata.timestamp,
            StreamEvent::GraphDeleted { metadata, .. } => metadata.timestamp,
            StreamEvent::SparqlUpdate { metadata, .. } => metadata.timestamp,
            StreamEvent::TransactionBegin { metadata, .. } => metadata.timestamp,
            StreamEvent::TransactionCommit { metadata, .. } => metadata.timestamp,
            StreamEvent::TransactionAbort { metadata, .. } => metadata.timestamp,
            StreamEvent::SchemaChanged { metadata, .. } => metadata.timestamp,
            StreamEvent::Heartbeat { timestamp, .. } => *timestamp,
            StreamEvent::QueryResultAdded { metadata, .. } => metadata.timestamp,
            StreamEvent::QueryResultRemoved { metadata, .. } => metadata.timestamp,
            StreamEvent::QueryCompleted { metadata, .. } => metadata.timestamp,
            StreamEvent::GraphMetadataUpdated { metadata, .. } => metadata.timestamp,
            StreamEvent::GraphPermissionsChanged { metadata, .. } => metadata.timestamp,
            StreamEvent::GraphStatisticsUpdated { metadata, .. } => metadata.timestamp,
            StreamEvent::GraphRenamed { metadata, .. } => metadata.timestamp,
            StreamEvent::GraphMerged { metadata, .. } => metadata.timestamp,
            StreamEvent::GraphSplit { metadata, .. } => metadata.timestamp,
            StreamEvent::SchemaDefinitionAdded { metadata, .. } => metadata.timestamp,
            StreamEvent::SchemaDefinitionRemoved { metadata, .. } => metadata.timestamp,
            StreamEvent::SchemaDefinitionModified { metadata, .. } => metadata.timestamp,
            StreamEvent::OntologyImported { metadata, .. } => metadata.timestamp,
            StreamEvent::OntologyRemoved { metadata, .. } => metadata.timestamp,
            StreamEvent::ConstraintAdded { metadata, .. } => metadata.timestamp,
            StreamEvent::ConstraintRemoved { metadata, .. } => metadata.timestamp,
            StreamEvent::ConstraintViolated { metadata, .. } => metadata.timestamp,
            StreamEvent::IndexCreated { metadata, .. } => metadata.timestamp,
            StreamEvent::IndexDropped { metadata, .. } => metadata.timestamp,
            StreamEvent::IndexRebuilt { metadata, .. } => metadata.timestamp,
            StreamEvent::SchemaUpdated { metadata, .. } => metadata.timestamp,
            StreamEvent::ShapeAdded { metadata, .. } => metadata.timestamp,
            StreamEvent::ShapeUpdated { metadata, .. } => metadata.timestamp,
            StreamEvent::ShapeRemoved { metadata, .. } => metadata.timestamp,
            StreamEvent::ShapeModified { metadata, .. } => metadata.timestamp,
            StreamEvent::ShapeValidationStarted { metadata, .. } => metadata.timestamp,
            StreamEvent::ShapeValidationCompleted { metadata, .. } => metadata.timestamp,
            StreamEvent::ShapeViolationDetected { metadata, .. } => metadata.timestamp,
            StreamEvent::ErrorOccurred { metadata, .. } => metadata.timestamp,
        }
    }
}

/// Event metadata for tracking and provenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetadata {
    /// Unique event identifier
    pub event_id: String,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Source identification
    pub source: String,
    /// User/session that triggered the event
    pub user: Option<String>,
    /// Operation context
    pub context: Option<String>,
    /// Causality tracking - event that caused this event
    pub caused_by: Option<String>,
    /// Event version for schema evolution
    pub version: String,
    /// Custom properties
    pub properties: HashMap<String, String>,
    /// Checksum for integrity
    pub checksum: Option<String>,
}

impl Default for EventMetadata {
    fn default() -> Self {
        Self {
            event_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            source: "default".to_string(),
            user: None,
            context: None,
            caused_by: None,
            version: "1.0".to_string(),
            properties: HashMap::new(),
            checksum: None,
        }
    }
}

/// SPARQL operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SparqlOperationType {
    Insert,
    Delete,
    Update,
    Load,
    Clear,
    Create,
    Drop,
    Copy,
    Move,
    Add,
}

/// Transaction isolation levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

/// Schema types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaType {
    Ontology,
    Vocabulary,
    Constraint,
    Rule,
}

/// Schema change types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaChangeType {
    Added,
    Modified,
    Removed,
    Versioned,
}

/// Stream event types for classification and filtering
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum StreamEventType {
    TripleAdded,
    TripleRemoved,
    QuadAdded,
    QuadRemoved,
    GraphCreated,
    GraphCleared,
    GraphDeleted,
    GraphMetadataUpdated,
    GraphPermissionsChanged,
    GraphStatisticsUpdated,
    GraphRenamed,
    GraphMerged,
    GraphSplit,
    SparqlUpdate,
    TransactionBegin,
    TransactionCommit,
    TransactionAbort,
    SchemaChanged,
    SchemaDefinitionAdded,
    SchemaDefinitionRemoved,
    SchemaDefinitionModified,
    OntologyImported,
    OntologyRemoved,
    ConstraintAdded,
    ConstraintRemoved,
    ConstraintViolated,
    IndexCreated,
    IndexDropped,
    IndexRebuilt,
    SchemaUpdated,
    ShapeAdded,
    ShapeRemoved,
    ShapeModified,
    ShapeUpdated,
    ShapeValidationStarted,
    ShapeValidationCompleted,
    ShapeViolationDetected,
    QueryResultAdded,
    QueryResultRemoved,
    QueryCompleted,
    Heartbeat,
    ErrorOccurred,
}

/// Query result placeholder (to be imported from store integration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub query_id: String,
    pub bindings: HashMap<String, String>,
    pub execution_time: Duration,
}

impl StreamEvent {
    /// Get the metadata for this event
    pub fn metadata(&self) -> &EventMetadata {
        match self {
            StreamEvent::TripleAdded { metadata, .. } => metadata,
            StreamEvent::TripleRemoved { metadata, .. } => metadata,
            StreamEvent::QuadAdded { metadata, .. } => metadata,
            StreamEvent::QuadRemoved { metadata, .. } => metadata,
            StreamEvent::GraphCreated { metadata, .. } => metadata,
            StreamEvent::GraphCleared { metadata, .. } => metadata,
            StreamEvent::GraphDeleted { metadata, .. } => metadata,
            StreamEvent::GraphMetadataUpdated { metadata, .. } => metadata,
            StreamEvent::GraphPermissionsChanged { metadata, .. } => metadata,
            StreamEvent::GraphStatisticsUpdated { metadata, .. } => metadata,
            StreamEvent::GraphRenamed { metadata, .. } => metadata,
            StreamEvent::GraphMerged { metadata, .. } => metadata,
            StreamEvent::GraphSplit { metadata, .. } => metadata,
            StreamEvent::SparqlUpdate { metadata, .. } => metadata,
            StreamEvent::TransactionBegin { metadata, .. } => metadata,
            StreamEvent::TransactionCommit { metadata, .. } => metadata,
            StreamEvent::TransactionAbort { metadata, .. } => metadata,
            StreamEvent::SchemaChanged { metadata, .. } => metadata,
            StreamEvent::SchemaDefinitionAdded { metadata, .. } => metadata,
            StreamEvent::SchemaDefinitionRemoved { metadata, .. } => metadata,
            StreamEvent::SchemaDefinitionModified { metadata, .. } => metadata,
            StreamEvent::OntologyImported { metadata, .. } => metadata,
            StreamEvent::OntologyRemoved { metadata, .. } => metadata,
            StreamEvent::ConstraintAdded { metadata, .. } => metadata,
            StreamEvent::ConstraintRemoved { metadata, .. } => metadata,
            StreamEvent::ConstraintViolated { metadata, .. } => metadata,
            StreamEvent::IndexCreated { metadata, .. } => metadata,
            StreamEvent::IndexDropped { metadata, .. } => metadata,
            StreamEvent::IndexRebuilt { metadata, .. } => metadata,
            StreamEvent::SchemaUpdated { metadata, .. } => metadata,
            StreamEvent::ShapeAdded { metadata, .. } => metadata,
            StreamEvent::ShapeRemoved { metadata, .. } => metadata,
            StreamEvent::ShapeModified { metadata, .. } => metadata,
            StreamEvent::ShapeUpdated { metadata, .. } => metadata,
            StreamEvent::ShapeValidationStarted { metadata, .. } => metadata,
            StreamEvent::ShapeValidationCompleted { metadata, .. } => metadata,
            StreamEvent::ShapeViolationDetected { metadata, .. } => metadata,
            StreamEvent::QueryResultAdded { metadata, .. } => metadata,
            StreamEvent::QueryResultRemoved { metadata, .. } => metadata,
            StreamEvent::QueryCompleted { metadata, .. } => metadata,
            StreamEvent::Heartbeat { metadata, .. } => metadata,
            StreamEvent::ErrorOccurred { metadata, .. } => metadata,
        }
    }

    // Helper methods for creating specific event types

    /// Create a named graph metadata update event
    pub fn graph_metadata_updated(
        graph: String,
        metadata_type: String,
        old_value: Option<String>,
        new_value: String,
    ) -> Self {
        StreamEvent::GraphMetadataUpdated {
            graph,
            metadata_type,
            old_value,
            new_value,
            metadata: EventMetadata::default(),
        }
    }

    /// Create a graph permissions change event
    pub fn graph_permissions_changed(
        graph: String,
        permission_type: String,
        principal: String,
        granted: bool,
    ) -> Self {
        StreamEvent::GraphPermissionsChanged {
            graph,
            permission_type,
            principal,
            granted,
            metadata: EventMetadata::default(),
        }
    }

    /// Create a graph statistics update event
    pub fn graph_statistics_updated(graph: String, triple_count: u64, size_bytes: u64) -> Self {
        StreamEvent::GraphStatisticsUpdated {
            graph,
            triple_count,
            size_bytes,
            last_modified: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metadata: EventMetadata::default(),
        }
    }

    /// Create a schema definition added event
    pub fn schema_definition_added(
        schema_type: String,
        schema_uri: String,
        definition: String,
    ) -> Self {
        StreamEvent::SchemaDefinitionAdded {
            schema_type,
            schema_uri,
            definition,
            metadata: EventMetadata::default(),
        }
    }

    /// Create a schema definition modified event
    pub fn schema_definition_modified(
        schema_type: String,
        schema_uri: String,
        old_definition: String,
        new_definition: String,
    ) -> Self {
        StreamEvent::SchemaDefinitionModified {
            schema_type,
            schema_uri,
            old_definition,
            new_definition,
            metadata: EventMetadata::default(),
        }
    }

    /// Create an ontology import event
    pub fn ontology_imported(
        ontology_uri: String,
        version: Option<String>,
        import_method: String,
    ) -> Self {
        StreamEvent::OntologyImported {
            ontology_uri,
            version,
            import_method,
            metadata: EventMetadata::default(),
        }
    }

    /// Create a constraint violation event
    pub fn constraint_violated(
        constraint_type: String,
        target: String,
        violating_data: String,
        severity: String,
    ) -> Self {
        StreamEvent::ConstraintViolated {
            constraint_type,
            target,
            violating_data,
            severity,
            metadata: EventMetadata::default(),
        }
    }

    /// Create an index creation event
    pub fn index_created(
        index_name: String,
        index_type: String,
        target_properties: Vec<String>,
        graph: Option<String>,
    ) -> Self {
        StreamEvent::IndexCreated {
            index_name,
            index_type,
            target_properties,
            graph,
            metadata: EventMetadata::default(),
        }
    }

    /// Create a SHACL shape added event
    pub fn shape_added(
        shape_uri: String,
        shape_definition: String,
        target_class: Option<String>,
    ) -> Self {
        StreamEvent::ShapeAdded {
            shape_uri,
            shape_definition,
            target_class,
            metadata: EventMetadata::default(),
        }
    }

    /// Create a SHACL shape validation completed event
    pub fn shape_validation_completed(
        shape_uri: String,
        validation_id: String,
        success: bool,
        violation_count: u32,
        duration_ms: u64,
    ) -> Self {
        StreamEvent::ShapeValidationCompleted {
            shape_uri,
            validation_id,
            success,
            violation_count,
            duration_ms,
            metadata: EventMetadata::default(),
        }
    }

    /// Create a SHACL shape violation detected event
    pub fn shape_violation_detected(
        shape_uri: String,
        violation_path: String,
        violating_node: String,
        severity: String,
        message: String,
    ) -> Self {
        StreamEvent::ShapeViolationDetected {
            shape_uri,
            violation_path,
            violating_node,
            severity,
            message,
            metadata: EventMetadata::default(),
        }
    }

    /// Get the event category for classification
    pub fn category(&self) -> EventCategory {
        match self {
            StreamEvent::TripleAdded { .. }
            | StreamEvent::TripleRemoved { .. }
            | StreamEvent::QuadAdded { .. }
            | StreamEvent::QuadRemoved { .. } => EventCategory::Data,

            StreamEvent::GraphCreated { .. }
            | StreamEvent::GraphCleared { .. }
            | StreamEvent::GraphDeleted { .. }
            | StreamEvent::GraphMetadataUpdated { .. }
            | StreamEvent::GraphPermissionsChanged { .. }
            | StreamEvent::GraphStatisticsUpdated { .. }
            | StreamEvent::GraphRenamed { .. }
            | StreamEvent::GraphMerged { .. }
            | StreamEvent::GraphSplit { .. } => EventCategory::Graph,

            StreamEvent::TransactionBegin { .. }
            | StreamEvent::TransactionCommit { .. }
            | StreamEvent::TransactionAbort { .. } => EventCategory::Transaction,

            StreamEvent::SchemaChanged { .. }
            | StreamEvent::SchemaDefinitionAdded { .. }
            | StreamEvent::SchemaDefinitionRemoved { .. }
            | StreamEvent::SchemaDefinitionModified { .. }
            | StreamEvent::SchemaUpdated { .. }
            | StreamEvent::OntologyImported { .. }
            | StreamEvent::OntologyRemoved { .. }
            | StreamEvent::ConstraintAdded { .. }
            | StreamEvent::ConstraintRemoved { .. }
            | StreamEvent::ConstraintViolated { .. } => EventCategory::Schema,

            StreamEvent::IndexCreated { .. }
            | StreamEvent::IndexDropped { .. }
            | StreamEvent::IndexRebuilt { .. } => EventCategory::Index,

            StreamEvent::ShapeAdded { .. }
            | StreamEvent::ShapeRemoved { .. }
            | StreamEvent::ShapeModified { .. }
            | StreamEvent::ShapeUpdated { .. }
            | StreamEvent::ShapeValidationStarted { .. }
            | StreamEvent::ShapeValidationCompleted { .. }
            | StreamEvent::ShapeViolationDetected { .. } => EventCategory::Shape,

            StreamEvent::SparqlUpdate { .. } => EventCategory::Query,

            StreamEvent::QueryResultAdded { .. }
            | StreamEvent::QueryResultRemoved { .. }
            | StreamEvent::QueryCompleted { .. } => EventCategory::Query,

            StreamEvent::Heartbeat { .. } => EventCategory::Data,
            
            StreamEvent::ErrorOccurred { .. } => EventCategory::Data,
        }
    }

    /// Get the specific event type for filtering and classification
    pub fn event_type(&self) -> StreamEventType {
        match self {
            StreamEvent::TripleAdded { .. } => StreamEventType::TripleAdded,
            StreamEvent::TripleRemoved { .. } => StreamEventType::TripleRemoved,
            StreamEvent::QuadAdded { .. } => StreamEventType::QuadAdded,
            StreamEvent::QuadRemoved { .. } => StreamEventType::QuadRemoved,
            StreamEvent::GraphCreated { .. } => StreamEventType::GraphCreated,
            StreamEvent::GraphCleared { .. } => StreamEventType::GraphCleared,
            StreamEvent::GraphDeleted { .. } => StreamEventType::GraphDeleted,
            StreamEvent::GraphMetadataUpdated { .. } => StreamEventType::GraphMetadataUpdated,
            StreamEvent::GraphPermissionsChanged { .. } => StreamEventType::GraphPermissionsChanged,
            StreamEvent::GraphStatisticsUpdated { .. } => StreamEventType::GraphStatisticsUpdated,
            StreamEvent::GraphRenamed { .. } => StreamEventType::GraphRenamed,
            StreamEvent::GraphMerged { .. } => StreamEventType::GraphMerged,
            StreamEvent::GraphSplit { .. } => StreamEventType::GraphSplit,
            StreamEvent::SparqlUpdate { .. } => StreamEventType::SparqlUpdate,
            StreamEvent::TransactionBegin { .. } => StreamEventType::TransactionBegin,
            StreamEvent::TransactionCommit { .. } => StreamEventType::TransactionCommit,
            StreamEvent::TransactionAbort { .. } => StreamEventType::TransactionAbort,
            StreamEvent::SchemaChanged { .. } => StreamEventType::SchemaChanged,
            StreamEvent::SchemaDefinitionAdded { .. } => StreamEventType::SchemaDefinitionAdded,
            StreamEvent::SchemaDefinitionRemoved { .. } => StreamEventType::SchemaDefinitionRemoved,
            StreamEvent::SchemaDefinitionModified { .. } => {
                StreamEventType::SchemaDefinitionModified
            }
            StreamEvent::OntologyImported { .. } => StreamEventType::OntologyImported,
            StreamEvent::OntologyRemoved { .. } => StreamEventType::OntologyRemoved,
            StreamEvent::ConstraintAdded { .. } => StreamEventType::ConstraintAdded,
            StreamEvent::ConstraintRemoved { .. } => StreamEventType::ConstraintRemoved,
            StreamEvent::ConstraintViolated { .. } => StreamEventType::ConstraintViolated,
            StreamEvent::IndexCreated { .. } => StreamEventType::IndexCreated,
            StreamEvent::IndexDropped { .. } => StreamEventType::IndexDropped,
            StreamEvent::IndexRebuilt { .. } => StreamEventType::IndexRebuilt,
            StreamEvent::SchemaUpdated { .. } => StreamEventType::SchemaUpdated,
            StreamEvent::ShapeAdded { .. } => StreamEventType::ShapeAdded,
            StreamEvent::ShapeRemoved { .. } => StreamEventType::ShapeRemoved,
            StreamEvent::ShapeModified { .. } => StreamEventType::ShapeModified,
            StreamEvent::ShapeUpdated { .. } => StreamEventType::ShapeUpdated,
            StreamEvent::ShapeValidationStarted { .. } => StreamEventType::ShapeValidationStarted,
            StreamEvent::ShapeValidationCompleted { .. } => {
                StreamEventType::ShapeValidationCompleted
            }
            StreamEvent::ShapeViolationDetected { .. } => StreamEventType::ShapeViolationDetected,
            StreamEvent::QueryResultAdded { .. } => StreamEventType::QueryResultAdded,
            StreamEvent::QueryResultRemoved { .. } => StreamEventType::QueryResultRemoved,
            StreamEvent::QueryCompleted { .. } => StreamEventType::QueryCompleted,
            StreamEvent::Heartbeat { .. } => StreamEventType::Heartbeat,
            StreamEvent::ErrorOccurred { .. } => StreamEventType::ErrorOccurred,
        }
    }

    /// Check if this event affects a specific graph
    pub fn affects_graph(&self, target_graph: &str) -> bool {
        match self {
            StreamEvent::TripleAdded { graph, .. } | StreamEvent::TripleRemoved { graph, .. } => {
                graph.as_ref().map_or(false, |g| g == target_graph)
            }
            StreamEvent::QuadAdded { graph, .. } | StreamEvent::QuadRemoved { graph, .. } => {
                graph == target_graph
            }
            StreamEvent::GraphCreated { graph, .. }
            | StreamEvent::GraphDeleted { graph, .. }
            | StreamEvent::GraphMetadataUpdated { graph, .. }
            | StreamEvent::GraphPermissionsChanged { graph, .. }
            | StreamEvent::GraphStatisticsUpdated { graph, .. } => graph == target_graph,
            StreamEvent::GraphCleared { graph, .. } => {
                graph.as_ref().map_or(true, |g| g == target_graph)
            }
            StreamEvent::GraphRenamed {
                old_name, new_name, ..
            } => old_name == target_graph || new_name == target_graph,
            StreamEvent::GraphMerged {
                source_graphs,
                target_graph: target,
                ..
            } => source_graphs.contains(&target_graph.to_string()) || target == target_graph,
            StreamEvent::GraphSplit {
                source_graph,
                target_graphs,
                ..
            } => source_graph == target_graph || target_graphs.contains(&target_graph.to_string()),
            StreamEvent::IndexCreated { graph, .. } => {
                graph.as_ref().map_or(false, |g| g == target_graph)
            }
            StreamEvent::ShapeValidationStarted {
                target_graph: shape_target,
                ..
            } => shape_target.as_ref().map_or(false, |g| g == target_graph),
            _ => false,
        }
    }

    /// Get the priority level of this event
    pub fn priority(&self) -> EventPriority {
        match self {
            StreamEvent::ConstraintViolated { severity, .. } => match severity.as_str() {
                "error" => EventPriority::High,
                "warning" => EventPriority::Medium,
                _ => EventPriority::Low,
            },
            StreamEvent::ShapeViolationDetected { severity, .. } => match severity.as_str() {
                "error" => EventPriority::High,
                "warning" => EventPriority::Medium,
                _ => EventPriority::Low,
            },
            StreamEvent::TransactionAbort { .. } | StreamEvent::GraphDeleted { .. } => {
                EventPriority::High
            }
            StreamEvent::IndexDropped { .. }
            | StreamEvent::OntologyRemoved { .. }
            | StreamEvent::SchemaDefinitionRemoved { .. } => EventPriority::Medium,
            _ => EventPriority::Low,
        }
    }
}

/// Event categories for classification and filtering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventCategory {
    Data,        // Triple/Quad operations
    Graph,       // Graph management
    Transaction, // Transaction control
    Schema,      // Schema and ontology changes
    Index,       // Index management
    Shape,       // SHACL shapes and validation
    Query,       // SPARQL operations
}

/// Event priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EventPriority {
    Low,
    Medium,
    High,
    Critical,
}

impl EventCategory {
    /// Get all available categories
    pub fn all() -> Vec<EventCategory> {
        vec![
            EventCategory::Data,
            EventCategory::Graph,
            EventCategory::Transaction,
            EventCategory::Schema,
            EventCategory::Index,
            EventCategory::Shape,
            EventCategory::Query,
        ]
    }

    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            EventCategory::Data => "Triple and quad data operations",
            EventCategory::Graph => "Named graph management operations",
            EventCategory::Transaction => "Transaction control operations",
            EventCategory::Schema => "Schema and ontology changes",
            EventCategory::Index => "Index management operations",
            EventCategory::Shape => "SHACL shape definition and validation",
            EventCategory::Query => "SPARQL query and update operations",
        }
    }
}
