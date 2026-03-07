//! # Event Sourcing Framework
//!
//! Complete event sourcing implementation for OxiRS Stream providing event storage,
//! replay capabilities, snapshots, and temporal queries. This forms the foundation
//! for CQRS patterns and enables advanced temporal analytics.

pub mod rdf_store_mod;
mod simple;
mod store;

#[cfg(test)]
mod simple_tests;
#[cfg(test)]
mod tests;

// Re-export all public types from sub-modules
pub use rdf_store_mod as rdf_store;
pub use simple::{
    EventStreamIter, ProjectionRunner, SimpleEvent, SimpleEventBus, SimpleEventHandler,
    SimpleEventStore, SimpleSnapshot, SimpleSnapshotStore,
};
pub use store::{EventIndexes, EventMetadataAccessor, EventStore, PersistenceManager};

use crate::StreamEvent;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use uuid::Uuid;

/// Event store configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventStoreConfig {
    /// Maximum events to keep in memory
    pub max_memory_events: usize,
    /// Enable persistent storage
    pub enable_persistence: bool,
    /// Persistence backend type
    pub persistence_backend: PersistenceBackend,
    /// Snapshot configuration
    pub snapshot_config: SnapshotConfig,
    /// Retention policy
    pub retention_policy: RetentionPolicy,
    /// Indexing configuration
    pub indexing_config: IndexingConfig,
    /// Enable compression for stored events
    pub enable_compression: bool,
    /// Batch size for persistence operations
    pub persistence_batch_size: usize,
}

impl Default for EventStoreConfig {
    fn default() -> Self {
        Self {
            max_memory_events: 1_000_000,
            enable_persistence: true,
            persistence_backend: PersistenceBackend::FileSystem {
                base_path: "/tmp/oxirs-event-store".to_string(),
            },
            snapshot_config: SnapshotConfig::default(),
            retention_policy: RetentionPolicy::default(),
            indexing_config: IndexingConfig::default(),
            enable_compression: true,
            persistence_batch_size: 1000,
        }
    }
}

/// Persistence backend options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PersistenceBackend {
    /// File system based storage
    FileSystem { base_path: String },
    /// Database storage
    Database { connection_string: String },
    /// S3-compatible object storage
    ObjectStorage {
        endpoint: String,
        bucket: String,
        access_key: String,
        secret_key: String,
    },
    /// In-memory only (no persistence)
    Memory,
}

/// Snapshot configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotConfig {
    /// Enable automatic snapshots
    pub enable_snapshots: bool,
    /// Snapshot interval (number of events)
    pub snapshot_interval: usize,
    /// Maximum snapshots to keep
    pub max_snapshots: usize,
    /// Snapshot compression
    pub compress_snapshots: bool,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            enable_snapshots: true,
            snapshot_interval: 10000,
            max_snapshots: 10,
            compress_snapshots: true,
        }
    }
}

/// Event retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Maximum age of events to keep
    pub max_age: Option<ChronoDuration>,
    /// Maximum number of events to keep
    pub max_events: Option<u64>,
    /// Archive old events instead of deleting
    pub enable_archiving: bool,
    /// Archive backend
    pub archive_backend: Option<PersistenceBackend>,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_age: Some(ChronoDuration::days(365)), // 1 year
            max_events: Some(10_000_000),             // 10M events
            enable_archiving: true,
            archive_backend: None,
        }
    }
}

/// Indexing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingConfig {
    /// Enable event type indexing
    pub index_by_event_type: bool,
    /// Enable timestamp indexing
    pub index_by_timestamp: bool,
    /// Enable source indexing
    pub index_by_source: bool,
    /// Enable custom field indexing
    pub custom_indexes: Vec<CustomIndex>,
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            index_by_event_type: true,
            index_by_timestamp: true,
            index_by_source: true,
            custom_indexes: Vec::new(),
        }
    }
}

/// Custom index definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomIndex {
    /// Index name
    pub name: String,
    /// Field path to index
    pub field_path: String,
    /// Index type
    pub index_type: IndexType,
}

/// Index type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    /// Hash index for exact matches
    Hash,
    /// B-tree index for range queries
    BTree,
    /// Full-text search index
    FullText,
}

/// Stored event with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredEvent {
    /// Unique event ID
    pub event_id: Uuid,
    /// Event sequence number (global order)
    pub sequence_number: u64,
    /// Stream ID (for grouping related events)
    pub stream_id: String,
    /// Event version within the stream
    pub stream_version: u64,
    /// Original event data
    pub event_data: StreamEvent,
    /// Storage timestamp
    pub stored_at: DateTime<Utc>,
    /// Storage metadata
    pub storage_metadata: StorageMetadata,
}

/// Storage metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageMetadata {
    /// Checksum for integrity verification
    pub checksum: String,
    /// Compressed size (if compressed)
    pub compressed_size: Option<usize>,
    /// Original size
    pub original_size: usize,
    /// Storage location
    pub storage_location: String,
    /// Persistence status
    pub persistence_status: PersistenceStatus,
}

/// Persistence status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PersistenceStatus {
    /// Only in memory
    InMemory,
    /// Persisted to disk
    Persisted,
    /// Archived to long-term storage
    Archived,
    /// Failed to persist
    Failed { error: String },
}

/// Event stream snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventSnapshot {
    /// Snapshot ID
    pub snapshot_id: Uuid,
    /// Stream ID
    pub stream_id: String,
    /// Stream version at snapshot time
    pub stream_version: u64,
    /// Sequence number at snapshot time
    pub sequence_number: u64,
    /// Snapshot timestamp
    pub created_at: DateTime<Utc>,
    /// Aggregated state data
    pub state_data: Vec<u8>,
    /// Snapshot metadata
    pub metadata: SnapshotMetadata,
}

/// Snapshot metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    /// Compression algorithm used
    pub compression: Option<String>,
    /// Original state size
    pub original_size: usize,
    /// Compressed size
    pub compressed_size: usize,
    /// Checksum for integrity
    pub checksum: String,
}

/// Query criteria for event retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventQuery {
    /// Stream ID filter
    pub stream_id: Option<String>,
    /// Event type filter
    pub event_types: Option<Vec<String>>,
    /// Time range filter
    pub time_range: Option<TimeRange>,
    /// Sequence number range
    pub sequence_range: Option<SequenceRange>,
    /// Source filter
    pub source: Option<String>,
    /// Custom field filters
    pub custom_filters: HashMap<String, String>,
    /// Maximum number of events to return
    pub limit: Option<usize>,
    /// Ordering preference
    pub order: QueryOrder,
}

/// Time range for queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

/// Sequence number range for queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceRange {
    pub start: u64,
    pub end: u64,
}

/// Query ordering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryOrder {
    /// Ascending by sequence number
    SequenceAsc,
    /// Descending by sequence number
    SequenceDesc,
    /// Ascending by timestamp
    TimestampAsc,
    /// Descending by timestamp
    TimestampDesc,
}

/// Event sourcing statistics
#[derive(Debug, Default)]
pub struct EventSourcingStats {
    pub total_events_stored: AtomicU64,
    pub total_events_retrieved: AtomicU64,
    pub snapshots_created: AtomicU64,
    pub events_archived: AtomicU64,
    pub persistence_operations: AtomicU64,
    pub failed_operations: AtomicU64,
    pub memory_usage_bytes: AtomicU64,
    pub disk_usage_bytes: AtomicU64,
    pub average_store_latency_ms: AtomicU64,
    pub average_retrieve_latency_ms: AtomicU64,
}

/// EventStore trait for abstracting event storage
#[async_trait::async_trait]
pub trait EventStoreTrait: Send + Sync {
    async fn store_event(
        &self,
        stream_id: String,
        event: StreamEvent,
    ) -> anyhow::Result<StoredEvent>;
    async fn query_events(&self, query: EventQuery) -> anyhow::Result<Vec<StoredEvent>>;
    async fn get_stream_events(
        &self,
        stream_id: &str,
        from_version: Option<u64>,
    ) -> anyhow::Result<Vec<StoredEvent>>;
    async fn replay_from_timestamp(
        &self,
        timestamp: DateTime<Utc>,
    ) -> anyhow::Result<Vec<StoredEvent>>;
    async fn get_latest_snapshot(&self, stream_id: &str) -> anyhow::Result<Option<EventSnapshot>>;
    async fn rebuild_stream_state(&self, stream_id: &str) -> anyhow::Result<Vec<u8>>;
    async fn append_events(
        &self,
        aggregate_id: &str,
        events: &[StreamEvent],
        expected_version: Option<u64>,
    ) -> anyhow::Result<u64>;
}

/// Event stream trait for streaming events
#[async_trait::async_trait]
pub trait EventStream: Send + Sync {
    async fn next_event(&mut self) -> Option<StoredEvent>;
    async fn has_events(&self) -> bool;
    async fn read_events_from_position(
        &self,
        position: u64,
        max_events: usize,
    ) -> anyhow::Result<Vec<StoredEvent>>;
}

/// Snapshot store trait for managing snapshots
#[async_trait::async_trait]
pub trait SnapshotStore: Send + Sync {
    async fn store_snapshot(&self, snapshot: EventSnapshot) -> anyhow::Result<()>;
    async fn get_snapshot(
        &self,
        stream_id: &str,
        version: Option<u64>,
    ) -> anyhow::Result<Option<EventSnapshot>>;
    async fn list_snapshots(&self, stream_id: &str) -> anyhow::Result<Vec<EventSnapshot>>;
}

/// Persistence operation
#[derive(Debug, Clone)]
pub enum PersistenceOperation {
    /// Store event
    StoreEvent(Box<StoredEvent>),
    /// Store snapshot
    StoreSnapshot(EventSnapshot),
    /// Archive events
    ArchiveEvents(Vec<StoredEvent>),
    /// Delete events
    DeleteEvents(Vec<u64>),
}

/// Persistence statistics
#[derive(Debug, Default)]
pub struct PersistenceStats {
    pub operations_queued: AtomicU64,
    pub operations_completed: AtomicU64,
    pub operations_failed: AtomicU64,
    pub bytes_written: AtomicU64,
    pub bytes_read: AtomicU64,
}
