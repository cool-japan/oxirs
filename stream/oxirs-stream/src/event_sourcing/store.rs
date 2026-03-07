//! EventStore, EventIndexes, PersistenceManager implementations and the
//! `EventMetadataAccessor` helper trait.

use super::{
    EventQuery, EventSnapshot, EventSourcingStats, EventStoreConfig, EventStoreTrait,
    PersistenceBackend, PersistenceOperation, PersistenceStats, PersistenceStatus, QueryOrder,
    SnapshotMetadata, StorageMetadata, StoredEvent, TimeRange,
};
use crate::{EventMetadata, StreamEvent};
use anyhow::Result;
use chrono::{DateTime, Utc};
use std::collections::VecDeque;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock, Semaphore};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Main event store implementation
pub struct EventStore {
    /// Configuration
    config: EventStoreConfig,
    /// In-memory event storage
    memory_events: Arc<RwLock<BTreeMap<u64, StoredEvent>>>,
    /// Stream version tracking
    stream_versions: Arc<RwLock<HashMap<String, u64>>>,
    /// Next sequence number
    next_sequence: Arc<AtomicU64>,
    /// Event indexes
    indexes: Arc<EventIndexes>,
    /// Snapshots
    snapshots: Arc<RwLock<HashMap<String, Vec<EventSnapshot>>>>,
    /// Persistence manager
    persistence_manager: Arc<PersistenceManager>,
    /// Statistics
    stats: Arc<EventSourcingStats>,
    /// Operation semaphore
    operation_semaphore: Arc<Semaphore>,
}

/// Event indexes for efficient querying
pub struct EventIndexes {
    /// Index by event type
    by_event_type: RwLock<HashMap<String, Vec<u64>>>,
    /// Index by timestamp
    by_timestamp: RwLock<BTreeMap<DateTime<Utc>, Vec<u64>>>,
    /// Index by source
    by_source: RwLock<HashMap<String, Vec<u64>>>,
    /// Index by stream ID
    by_stream: RwLock<HashMap<String, Vec<u64>>>,
    /// Custom indexes
    custom_indexes: RwLock<HashMap<String, HashMap<String, Vec<u64>>>>,
}

/// Persistence manager for durable storage
pub struct PersistenceManager {
    /// Backend configuration
    backend: PersistenceBackend,
    /// Pending operations queue
    pending_operations: Arc<Mutex<VecDeque<PersistenceOperation>>>,
    /// Persistence statistics
    pub(crate) stats: Arc<PersistenceStats>,
}

impl EventStore {
    /// Create a new event store
    pub fn new(config: EventStoreConfig) -> Self {
        let persistence_manager =
            Arc::new(PersistenceManager::new(config.persistence_backend.clone()));

        Self {
            config,
            memory_events: Arc::new(RwLock::new(BTreeMap::new())),
            stream_versions: Arc::new(RwLock::new(HashMap::new())),
            next_sequence: Arc::new(AtomicU64::new(1)),
            indexes: Arc::new(EventIndexes::new()),
            snapshots: Arc::new(RwLock::new(HashMap::new())),
            persistence_manager,
            stats: Arc::new(EventSourcingStats::default()),
            operation_semaphore: Arc::new(Semaphore::new(1000)), // Max 1000 concurrent operations
        }
    }

    /// Store an event in the event store
    pub async fn store_event(&self, stream_id: String, event: StreamEvent) -> Result<StoredEvent> {
        let _permit = self.operation_semaphore.acquire().await?;
        let start_time = Instant::now();

        // Generate sequence number and stream version
        let sequence_number = self.next_sequence.fetch_add(1, Ordering::SeqCst);
        let stream_version = {
            let mut versions = self.stream_versions.write().await;
            let version = versions.get(&stream_id).unwrap_or(&0) + 1;
            versions.insert(stream_id.clone(), version);
            version
        };

        // Create stored event
        let checksum = self.calculate_checksum(&event)?;
        let original_size = self.estimate_size(&event);
        let stored_event = StoredEvent {
            event_id: Uuid::new_v4(),
            sequence_number,
            stream_id: stream_id.clone(),
            stream_version,
            event_data: event,
            stored_at: Utc::now(),
            storage_metadata: StorageMetadata {
                checksum,
                compressed_size: None,
                original_size,
                storage_location: format!("memory:{sequence_number}"),
                persistence_status: PersistenceStatus::InMemory,
            },
        };

        // Store in memory
        {
            let mut memory_events = self.memory_events.write().await;
            memory_events.insert(sequence_number, stored_event.clone());

            // Evict old events if needed
            if memory_events.len() > self.config.max_memory_events {
                let to_remove: Vec<u64> = memory_events
                    .keys()
                    .take(memory_events.len() - self.config.max_memory_events)
                    .cloned()
                    .collect();

                for seq in to_remove {
                    memory_events.remove(&seq);
                }
            }
        }

        // Update indexes
        self.indexes.add_event(&stored_event).await?;

        // Queue for persistence if enabled
        if self.config.enable_persistence {
            self.persistence_manager
                .queue_operation(PersistenceOperation::StoreEvent(Box::new(
                    stored_event.clone(),
                )))
                .await?;
        }

        // Check if snapshot is needed
        if self.config.snapshot_config.enable_snapshots
            && stream_version % self.config.snapshot_config.snapshot_interval as u64 == 0
        {
            self.create_snapshot(&stream_id, stream_version).await?;
        }

        // Update statistics
        self.stats
            .total_events_stored
            .fetch_add(1, Ordering::Relaxed);
        let store_latency = start_time.elapsed();
        self.stats
            .average_store_latency_ms
            .store(store_latency.as_millis() as u64, Ordering::Relaxed);

        info!(
            "Stored event {} for stream {} (seq: {}, version: {})",
            stored_event.event_id, stream_id, sequence_number, stream_version
        );

        Ok(stored_event)
    }

    /// Retrieve events by query
    pub async fn query_events(&self, query: EventQuery) -> Result<Vec<StoredEvent>> {
        let _permit = self.operation_semaphore.acquire().await?;
        let start_time = Instant::now();

        let candidate_sequences = self.indexes.find_matching_sequences(&query).await?;
        let mut results = Vec::new();

        let memory_events = self.memory_events.read().await;
        for &sequence in &candidate_sequences {
            if let Some(stored_event) = memory_events.get(&sequence) {
                if self.matches_query(stored_event, &query) {
                    results.push(stored_event.clone());

                    if let Some(limit) = query.limit {
                        if results.len() >= limit {
                            break;
                        }
                    }
                }
            }
        }

        // Sort results based on query order
        self.sort_results(&mut results, &query.order);

        // Update statistics
        self.stats
            .total_events_retrieved
            .fetch_add(results.len() as u64, Ordering::Relaxed);
        let retrieve_latency = start_time.elapsed();
        self.stats
            .average_retrieve_latency_ms
            .store(retrieve_latency.as_millis() as u64, Ordering::Relaxed);

        debug!(
            "Query returned {} events in {:?}",
            results.len(),
            retrieve_latency
        );

        Ok(results)
    }

    /// Get events for a specific stream
    pub async fn get_stream_events(
        &self,
        stream_id: &str,
        from_version: Option<u64>,
    ) -> Result<Vec<StoredEvent>> {
        let query = EventQuery {
            stream_id: Some(stream_id.to_string()),
            event_types: None,
            time_range: None,
            sequence_range: None,
            source: None,
            custom_filters: HashMap::new(),
            limit: None,
            order: QueryOrder::SequenceAsc,
        };

        let mut events = self.query_events(query).await?;

        if let Some(from_version) = from_version {
            events.retain(|e| e.stream_version >= from_version);
        }

        Ok(events)
    }

    /// Replay events from a specific point in time
    pub async fn replay_from_timestamp(
        &self,
        timestamp: DateTime<Utc>,
    ) -> Result<Vec<StoredEvent>> {
        let query = EventQuery {
            stream_id: None,
            event_types: None,
            time_range: Some(TimeRange {
                start: timestamp,
                end: Utc::now(),
            }),
            sequence_range: None,
            source: None,
            custom_filters: HashMap::new(),
            limit: None,
            order: QueryOrder::SequenceAsc,
        };

        self.query_events(query).await
    }

    /// Create a snapshot for a stream
    async fn create_snapshot(&self, stream_id: &str, stream_version: u64) -> Result<EventSnapshot> {
        let events = self.get_stream_events(stream_id, None).await?;

        // Aggregate state from events (simplified)
        let state_data = self.aggregate_events(&events)?;
        let compressed_data = self.compress_data(&state_data)?;

        let snapshot = EventSnapshot {
            snapshot_id: Uuid::new_v4(),
            stream_id: stream_id.to_string(),
            stream_version,
            sequence_number: events.last().map(|e| e.sequence_number).unwrap_or(0),
            created_at: Utc::now(),
            state_data: compressed_data.clone(),
            metadata: SnapshotMetadata {
                compression: Some("gzip".to_string()),
                original_size: state_data.len(),
                compressed_size: compressed_data.len(),
                checksum: self.calculate_data_checksum(&compressed_data)?,
            },
        };

        // Store snapshot
        {
            let mut snapshots = self.snapshots.write().await;
            let stream_snapshots = snapshots
                .entry(stream_id.to_string())
                .or_insert_with(Vec::new);
            stream_snapshots.push(snapshot.clone());

            // Keep only recent snapshots
            if stream_snapshots.len() > self.config.snapshot_config.max_snapshots {
                stream_snapshots.remove(0);
            }
        }

        // Queue for persistence
        if self.config.enable_persistence {
            self.persistence_manager
                .queue_operation(PersistenceOperation::StoreSnapshot(snapshot.clone()))
                .await?;
        }

        self.stats.snapshots_created.fetch_add(1, Ordering::Relaxed);
        info!(
            "Created snapshot {} for stream {} at version {}",
            snapshot.snapshot_id, stream_id, stream_version
        );

        Ok(snapshot)
    }

    /// Get the latest snapshot for a stream
    pub async fn get_latest_snapshot(&self, stream_id: &str) -> Result<Option<EventSnapshot>> {
        let snapshots = self.snapshots.read().await;
        if let Some(stream_snapshots) = snapshots.get(stream_id) {
            Ok(stream_snapshots.last().cloned())
        } else {
            Ok(None)
        }
    }

    /// Rebuild stream state from events and snapshots
    pub async fn rebuild_stream_state(&self, stream_id: &str) -> Result<Vec<u8>> {
        // Get latest snapshot
        if let Some(snapshot) = self.get_latest_snapshot(stream_id).await? {
            // Get events after snapshot
            let events = self
                .get_stream_events(stream_id, Some(snapshot.stream_version + 1))
                .await?;

            // Start with snapshot state
            let mut state = self.decompress_data(&snapshot.state_data)?;

            // Apply subsequent events
            for event in events {
                state = self.apply_event_to_state(state, &event.event_data)?;
            }

            Ok(state)
        } else {
            // No snapshot, rebuild from all events
            let events = self.get_stream_events(stream_id, None).await?;
            let aggregated = self.aggregate_events(&events)?;
            Ok(aggregated)
        }
    }

    /// Check if an event matches the query criteria
    fn matches_query(&self, event: &StoredEvent, query: &EventQuery) -> bool {
        // Stream ID filter
        if let Some(ref stream_id) = query.stream_id {
            if &event.stream_id != stream_id {
                return false;
            }
        }

        // Event type filter
        if let Some(ref event_types) = query.event_types {
            let event_type = format!("{:?}", std::mem::discriminant(&event.event_data));
            if !event_types.contains(&event_type) {
                return false;
            }
        }

        // Time range filter
        if let Some(ref time_range) = query.time_range {
            let event_time = event.event_data.metadata().timestamp;
            if event_time < time_range.start || event_time > time_range.end {
                return false;
            }
        }

        // Sequence range filter
        if let Some(ref seq_range) = query.sequence_range {
            if event.sequence_number < seq_range.start || event.sequence_number > seq_range.end {
                return false;
            }
        }

        // Source filter
        if let Some(ref source) = query.source {
            if &event.event_data.metadata().source != source {
                return false;
            }
        }

        true
    }

    /// Sort results based on query order
    fn sort_results(&self, results: &mut [StoredEvent], order: &QueryOrder) {
        match order {
            QueryOrder::SequenceAsc => {
                results.sort_by_key(|e| e.sequence_number);
            }
            QueryOrder::SequenceDesc => {
                results.sort_by_key(|e| std::cmp::Reverse(e.sequence_number));
            }
            QueryOrder::TimestampAsc => {
                results.sort_by_key(|e| e.event_data.metadata().timestamp);
            }
            QueryOrder::TimestampDesc => {
                results.sort_by_key(|e| std::cmp::Reverse(e.event_data.metadata().timestamp));
            }
        }
    }

    /// Calculate checksum for event
    fn calculate_checksum(&self, event: &StreamEvent) -> Result<String> {
        let serialized = serde_json::to_string(event)?;
        Ok(format!("{:x}", crc32fast::hash(serialized.as_bytes())))
    }

    /// Calculate checksum for data
    fn calculate_data_checksum(&self, data: &[u8]) -> Result<String> {
        Ok(format!("{:x}", crc32fast::hash(data)))
    }

    /// Estimate size of an event
    fn estimate_size(&self, event: &StreamEvent) -> usize {
        serde_json::to_string(event)
            .map(|s| s.len())
            .unwrap_or(1024)
    }

    /// Aggregate events into state data
    fn aggregate_events(&self, events: &[StoredEvent]) -> Result<Vec<u8>> {
        // Simplified aggregation - in real implementation, this would be domain-specific
        let aggregate = format!("Aggregated {} events", events.len());
        Ok(aggregate.into_bytes())
    }

    /// Apply an event to existing state
    fn apply_event_to_state(&self, mut state: Vec<u8>, _event: &StreamEvent) -> Result<Vec<u8>> {
        // Simplified state application
        state.extend_from_slice(b" +event");
        Ok(state)
    }

    /// Compress data
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        if self.config.enable_compression {
            use flate2::write::GzEncoder;
            use flate2::Compression;
            use std::io::Write;

            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(data)?;
            Ok(encoder.finish()?)
        } else {
            Ok(data.to_vec())
        }
    }

    /// Decompress data
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        if self.config.enable_compression {
            use flate2::read::GzDecoder;
            use std::io::Read;

            let mut decoder = GzDecoder::new(data);
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed)?;
            Ok(decompressed)
        } else {
            Ok(data.to_vec())
        }
    }

    /// Get event sourcing statistics
    pub fn get_stats(&self) -> super::EventSourcingStats {
        super::EventSourcingStats {
            total_events_stored: AtomicU64::new(
                self.stats.total_events_stored.load(Ordering::Relaxed),
            ),
            total_events_retrieved: AtomicU64::new(
                self.stats.total_events_retrieved.load(Ordering::Relaxed),
            ),
            snapshots_created: AtomicU64::new(self.stats.snapshots_created.load(Ordering::Relaxed)),
            events_archived: AtomicU64::new(self.stats.events_archived.load(Ordering::Relaxed)),
            persistence_operations: AtomicU64::new(
                self.stats.persistence_operations.load(Ordering::Relaxed),
            ),
            failed_operations: AtomicU64::new(self.stats.failed_operations.load(Ordering::Relaxed)),
            memory_usage_bytes: AtomicU64::new(
                self.stats.memory_usage_bytes.load(Ordering::Relaxed),
            ),
            disk_usage_bytes: AtomicU64::new(self.stats.disk_usage_bytes.load(Ordering::Relaxed)),
            average_store_latency_ms: AtomicU64::new(
                self.stats.average_store_latency_ms.load(Ordering::Relaxed),
            ),
            average_retrieve_latency_ms: AtomicU64::new(
                self.stats
                    .average_retrieve_latency_ms
                    .load(Ordering::Relaxed),
            ),
        }
    }
}

/// Implement the EventStoreTrait for the concrete EventStore
#[async_trait::async_trait]
impl EventStoreTrait for EventStore {
    async fn store_event(&self, stream_id: String, event: StreamEvent) -> Result<StoredEvent> {
        self.store_event(stream_id, event).await
    }

    async fn query_events(&self, query: EventQuery) -> Result<Vec<StoredEvent>> {
        self.query_events(query).await
    }

    async fn get_stream_events(
        &self,
        stream_id: &str,
        from_version: Option<u64>,
    ) -> Result<Vec<StoredEvent>> {
        self.get_stream_events(stream_id, from_version).await
    }

    async fn replay_from_timestamp(&self, timestamp: DateTime<Utc>) -> Result<Vec<StoredEvent>> {
        self.replay_from_timestamp(timestamp).await
    }

    async fn get_latest_snapshot(&self, stream_id: &str) -> Result<Option<EventSnapshot>> {
        self.get_latest_snapshot(stream_id).await
    }

    async fn rebuild_stream_state(&self, stream_id: &str) -> Result<Vec<u8>> {
        self.rebuild_stream_state(stream_id).await
    }

    async fn append_events(
        &self,
        aggregate_id: &str,
        events: &[StreamEvent],
        _expected_version: Option<u64>,
    ) -> Result<u64> {
        let mut last_version = 0u64;
        for event in events {
            let stored_event = self
                .store_event(aggregate_id.to_string(), event.clone())
                .await?;
            last_version = stored_event.stream_version;
        }
        Ok(last_version)
    }
}

impl Default for EventIndexes {
    fn default() -> Self {
        Self::new()
    }
}

impl EventIndexes {
    /// Create new event indexes
    pub fn new() -> Self {
        Self {
            by_event_type: RwLock::new(HashMap::new()),
            by_timestamp: RwLock::new(BTreeMap::new()),
            by_source: RwLock::new(HashMap::new()),
            by_stream: RwLock::new(HashMap::new()),
            custom_indexes: RwLock::new(HashMap::new()),
        }
    }

    /// Add an event to indexes
    pub async fn add_event(&self, event: &StoredEvent) -> Result<()> {
        let sequence = event.sequence_number;

        // Index by event type
        {
            let mut by_type = self.by_event_type.write().await;
            let event_type = format!("{:?}", std::mem::discriminant(&event.event_data));
            by_type
                .entry(event_type)
                .or_insert_with(Vec::new)
                .push(sequence);
        }

        // Index by timestamp
        {
            let mut by_timestamp = self.by_timestamp.write().await;
            let timestamp = event.event_data.metadata().timestamp;
            by_timestamp
                .entry(timestamp)
                .or_insert_with(Vec::new)
                .push(sequence);
        }

        // Index by source
        {
            let mut by_source = self.by_source.write().await;
            let source = &event.event_data.metadata().source;
            by_source
                .entry(source.clone())
                .or_insert_with(Vec::new)
                .push(sequence);
        }

        // Index by stream
        {
            let mut by_stream = self.by_stream.write().await;
            by_stream
                .entry(event.stream_id.clone())
                .or_insert_with(Vec::new)
                .push(sequence);
        }

        Ok(())
    }

    /// Find sequences matching query criteria
    pub async fn find_matching_sequences(&self, query: &EventQuery) -> Result<Vec<u64>> {
        let mut candidate_sequences = Vec::new();

        // Start with stream filter if specified
        if let Some(ref stream_id) = query.stream_id {
            let by_stream = self.by_stream.read().await;
            if let Some(sequences) = by_stream.get(stream_id) {
                candidate_sequences = sequences.clone();
            } else {
                return Ok(Vec::new()); // Stream not found
            }
        } else {
            // Get all sequences (this could be optimized)
            let by_stream = self.by_stream.read().await;
            for sequences in by_stream.values() {
                candidate_sequences.extend(sequences);
            }
        }

        // Apply other filters
        if let Some(ref event_types) = query.event_types {
            let by_type = self.by_event_type.read().await;
            let mut type_sequences: HashSet<u64> = HashSet::new();

            for event_type in event_types {
                if let Some(sequences) = by_type.get(event_type) {
                    type_sequences.extend(sequences);
                }
            }

            candidate_sequences.retain(|seq| type_sequences.contains(seq));
        }

        // Apply sequence range filter
        if let Some(ref seq_range) = query.sequence_range {
            candidate_sequences.retain(|&seq| seq >= seq_range.start && seq <= seq_range.end);
        }

        candidate_sequences.sort_unstable();
        Ok(candidate_sequences)
    }
}

impl PersistenceManager {
    /// Create new persistence manager
    pub fn new(backend: PersistenceBackend) -> Self {
        Self {
            backend,
            pending_operations: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(PersistenceStats::default()),
        }
    }

    /// Queue a persistence operation
    pub async fn queue_operation(&self, operation: PersistenceOperation) -> Result<()> {
        let mut queue = self.pending_operations.lock().await;
        queue.push_back(operation);
        self.stats.operations_queued.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Process pending persistence operations
    pub async fn process_pending_operations(&self) -> Result<()> {
        let operations: Vec<PersistenceOperation> = {
            let mut queue = self.pending_operations.lock().await;
            queue.drain(..).collect()
        };

        for operation in operations {
            match self.execute_operation(operation).await {
                Ok(_) => {
                    self.stats
                        .operations_completed
                        .fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => {
                    self.stats.operations_failed.fetch_add(1, Ordering::Relaxed);
                    error!("Persistence operation failed: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Execute a single persistence operation
    async fn execute_operation(&self, operation: PersistenceOperation) -> Result<()> {
        match &self.backend {
            PersistenceBackend::Memory => {
                // No-op for memory backend
                Ok(())
            }
            PersistenceBackend::FileSystem { base_path } => {
                self.execute_filesystem_operation(operation, base_path)
                    .await
            }
            _ => {
                // Other backends not implemented in this example
                warn!("Persistence backend not implemented: {:?}", self.backend);
                Ok(())
            }
        }
    }

    /// Execute filesystem persistence operation
    async fn execute_filesystem_operation(
        &self,
        operation: PersistenceOperation,
        _base_path: &str,
    ) -> Result<()> {
        match operation {
            PersistenceOperation::StoreEvent(_event) => {
                // Simulate file write
                tokio::time::sleep(Duration::from_millis(1)).await;
                self.stats.bytes_written.fetch_add(1024, Ordering::Relaxed);
            }
            PersistenceOperation::StoreSnapshot(_snapshot) => {
                // Simulate snapshot write
                tokio::time::sleep(Duration::from_millis(5)).await;
                self.stats.bytes_written.fetch_add(10240, Ordering::Relaxed);
            }
            _ => {
                // Other operations
            }
        }
        Ok(())
    }
}

// Helper trait for accessing metadata
pub trait EventMetadataAccessor {
    fn metadata(&self) -> &EventMetadata;
}

impl EventMetadataAccessor for StreamEvent {
    fn metadata(&self) -> &EventMetadata {
        match self {
            StreamEvent::TripleAdded { metadata, .. } => metadata,
            StreamEvent::TripleRemoved { metadata, .. } => metadata,
            StreamEvent::QuadAdded { metadata, .. } => metadata,
            StreamEvent::QuadRemoved { metadata, .. } => metadata,
            StreamEvent::GraphCreated { metadata, .. } => metadata,
            StreamEvent::GraphCleared { metadata, .. } => metadata,
            StreamEvent::GraphDeleted { metadata, .. } => metadata,
            StreamEvent::SparqlUpdate { metadata, .. } => metadata,
            StreamEvent::TransactionBegin { metadata, .. } => metadata,
            StreamEvent::TransactionCommit { metadata, .. } => metadata,
            StreamEvent::TransactionAbort { metadata, .. } => metadata,
            StreamEvent::SchemaChanged { metadata, .. } => metadata,
            StreamEvent::Heartbeat { metadata, .. } => metadata,
            StreamEvent::QueryResultAdded { metadata, .. } => metadata,
            StreamEvent::QueryResultRemoved { metadata, .. } => metadata,
            StreamEvent::QueryCompleted { metadata, .. } => metadata,
            StreamEvent::ErrorOccurred { metadata, .. } => metadata,
            _ => {
                // For unmatched event types, return a static reference
                use once_cell::sync::Lazy;
                static DEFAULT_METADATA: Lazy<EventMetadata> = Lazy::new(EventMetadata::default);
                &DEFAULT_METADATA
            }
        }
    }
}
