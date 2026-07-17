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
use serde::de::DeserializeOwned;
use std::collections::VecDeque;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::io::AsyncWriteExt;
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
    /// Create a new event store.
    ///
    /// This is fallible because construction may need to create the on-disk
    /// persistence directory (fail-fast if that isn't possible) and loads any
    /// previously persisted events/snapshots back into memory ("load-on-open")
    /// so a restart never silently drops durable history.
    pub async fn new(config: EventStoreConfig) -> Result<Self> {
        let persistence_manager =
            Arc::new(PersistenceManager::new(config.persistence_backend.clone())?);

        let memory_events = Arc::new(RwLock::new(BTreeMap::new()));
        let stream_versions = Arc::new(RwLock::new(HashMap::new()));
        let indexes = Arc::new(EventIndexes::new());
        let snapshots = Arc::new(RwLock::new(HashMap::new()));
        let next_sequence = Arc::new(AtomicU64::new(1));

        // Load-on-open: recover previously persisted events so eviction never
        // has to guess whether history already made it to durable storage.
        let loaded_events = persistence_manager.load_persisted_events().await?;
        let mut max_sequence = 0u64;
        {
            let mut mem = memory_events.write().await;
            let mut versions = stream_versions.write().await;
            for mut event in loaded_events {
                // These records came from durable storage, so they are safe to evict again.
                event.storage_metadata.persistence_status = PersistenceStatus::Persisted;
                max_sequence = max_sequence.max(event.sequence_number);
                let version_slot = versions.entry(event.stream_id.clone()).or_insert(0);
                if event.stream_version > *version_slot {
                    *version_slot = event.stream_version;
                }
                indexes.add_event(&event).await?;
                mem.insert(event.sequence_number, event);
            }
        }
        next_sequence.store(max_sequence + 1, Ordering::SeqCst);

        let loaded_snapshots = persistence_manager.load_persisted_snapshots().await?;
        {
            let mut snapshot_map = snapshots.write().await;
            for snapshot in loaded_snapshots {
                snapshot_map
                    .entry(snapshot.stream_id.clone())
                    .or_insert_with(Vec::new)
                    .push(snapshot);
            }
            for stream_snapshots in snapshot_map.values_mut() {
                stream_snapshots.sort_by_key(|s| s.stream_version);
            }
        }

        Ok(Self {
            config,
            memory_events,
            stream_versions,
            next_sequence,
            indexes,
            snapshots,
            persistence_manager,
            stats: Arc::new(EventSourcingStats::default()),
            operation_semaphore: Arc::new(Semaphore::new(1000)), // Max 1000 concurrent operations
        })
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
        let mut stored_event = StoredEvent {
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
        }

        // Update indexes
        self.indexes.add_event(&stored_event).await?;

        // Persist synchronously (not just queued) so we know the true durability
        // status of this event before it can ever be considered for eviction.
        if self.config.enable_persistence && self.persistence_manager.is_durable() {
            match self.persistence_manager.persist_event(&stored_event).await {
                Ok(()) => {
                    stored_event.storage_metadata.persistence_status = PersistenceStatus::Persisted;
                    self.stats
                        .persistence_operations
                        .fetch_add(1, Ordering::Relaxed);
                    let mut memory_events = self.memory_events.write().await;
                    if let Some(entry) = memory_events.get_mut(&sequence_number) {
                        entry.storage_metadata.persistence_status = PersistenceStatus::Persisted;
                    }
                }
                Err(e) => {
                    error!(
                        "Failed to persist event {} for stream {}: {}",
                        stored_event.event_id, stream_id, e
                    );
                    stored_event.storage_metadata.persistence_status = PersistenceStatus::Failed {
                        error: e.to_string(),
                    };
                    self.stats.failed_operations.fetch_add(1, Ordering::Relaxed);
                    let mut memory_events = self.memory_events.write().await;
                    if let Some(entry) = memory_events.get_mut(&sequence_number) {
                        entry.storage_metadata.persistence_status = PersistenceStatus::Failed {
                            error: e.to_string(),
                        };
                    }
                }
            }
        }

        // Evict old events, but only ones that are safe to lose from memory:
        // either persistence isn't expected for this store (Memory backend or
        // persistence disabled), or the event has actually made it to durable
        // storage. Events still pending/failed persistence are never evicted,
        // so "persisted" eviction never silently discards undurable data.
        {
            let mut memory_events = self.memory_events.write().await;
            if memory_events.len() > self.config.max_memory_events {
                let overflow = memory_events.len() - self.config.max_memory_events;
                let evictable: Vec<u64> = memory_events
                    .iter()
                    .filter(|(_, event)| self.can_evict(event))
                    .map(|(seq, _)| *seq)
                    .take(overflow)
                    .collect();

                for seq in evictable {
                    memory_events.remove(&seq);
                }
            }
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

        // Persist the snapshot synchronously; a snapshot write failure is a real
        // error the caller needs to know about rather than a silently dropped write.
        if self.config.enable_persistence && self.persistence_manager.is_durable() {
            self.persistence_manager
                .persist_snapshot(&snapshot)
                .await
                .map_err(|e| {
                    anyhow::anyhow!("Failed to persist snapshot for stream {stream_id}: {e}")
                })?;
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

    /// Whether an in-memory event is safe to evict: either this store never
    /// promised durability for it (persistence disabled, or a non-durable
    /// backend such as `Memory`), or it has actually been written to durable
    /// storage. Events that are still pending or failed persistence are kept
    /// in memory so eviction can never silently discard undurable data.
    fn can_evict(&self, event: &StoredEvent) -> bool {
        if !self.config.enable_persistence || !self.persistence_manager.is_durable() {
            return true;
        }
        matches!(
            event.storage_metadata.persistence_status,
            PersistenceStatus::Persisted | PersistenceStatus::Archived
        )
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
            oxiarc_deflate::gzip_compress(data, 6)
                .map_err(|e| anyhow::anyhow!("Gzip compression failed: {e}"))
        } else {
            Ok(data.to_vec())
        }
    }

    /// Decompress data
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        if self.config.enable_compression {
            oxiarc_deflate::gzip_decompress(data)
                .map_err(|e| anyhow::anyhow!("Gzip decompression failed: {e}"))
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
    /// Create new persistence manager.
    ///
    /// For the `FileSystem` backend this eagerly creates the base directory so
    /// construction fails fast (rather than failing later, silently, on the
    /// first write) if the path is unusable.
    pub fn new(backend: PersistenceBackend) -> Result<Self> {
        if let PersistenceBackend::FileSystem { base_path } = &backend {
            std::fs::create_dir_all(base_path).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to create event store persistence directory '{base_path}': {e}"
                )
            })?;
        }

        Ok(Self {
            backend,
            pending_operations: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(PersistenceStats::default()),
        })
    }

    /// Whether this backend actually durably persists data. `Memory` is an
    /// explicit "no persistence" backend, so events stored under it were never
    /// expected to survive a restart or be excluded from eviction.
    pub fn is_durable(&self) -> bool {
        matches!(self.backend, PersistenceBackend::FileSystem { .. })
    }

    /// Persist a single event immediately (append-only JSON-lines write with
    /// an fsync before returning), so the caller knows for certain whether the
    /// event actually reached durable storage.
    pub async fn persist_event(&self, event: &StoredEvent) -> Result<()> {
        match &self.backend {
            PersistenceBackend::Memory => Ok(()),
            PersistenceBackend::FileSystem { base_path } => {
                let line = serde_json::to_string(event)?;
                self.append_jsonl(base_path, "events.jsonl", line).await
            }
            PersistenceBackend::Database { .. } | PersistenceBackend::ObjectStorage { .. } => {
                Err(anyhow::anyhow!(
                    "Persistence backend {:?} is not implemented; use FileSystem or Memory",
                    self.backend
                ))
            }
        }
    }

    /// Persist a snapshot immediately (append-only JSON-lines write with an
    /// fsync before returning).
    pub async fn persist_snapshot(&self, snapshot: &EventSnapshot) -> Result<()> {
        match &self.backend {
            PersistenceBackend::Memory => Ok(()),
            PersistenceBackend::FileSystem { base_path } => {
                let line = serde_json::to_string(snapshot)?;
                self.append_jsonl(base_path, "snapshots.jsonl", line).await
            }
            PersistenceBackend::Database { .. } | PersistenceBackend::ObjectStorage { .. } => {
                Err(anyhow::anyhow!(
                    "Persistence backend {:?} is not implemented; use FileSystem or Memory",
                    self.backend
                ))
            }
        }
    }

    /// Load all previously persisted events (load-on-open recovery).
    pub async fn load_persisted_events(&self) -> Result<Vec<StoredEvent>> {
        match &self.backend {
            PersistenceBackend::FileSystem { base_path } => {
                let path = Path::new(base_path).join("events.jsonl");
                Self::read_jsonl(&path).await
            }
            _ => Ok(Vec::new()),
        }
    }

    /// Load all previously persisted snapshots (load-on-open recovery).
    pub async fn load_persisted_snapshots(&self) -> Result<Vec<EventSnapshot>> {
        match &self.backend {
            PersistenceBackend::FileSystem { base_path } => {
                let path = Path::new(base_path).join("snapshots.jsonl");
                Self::read_jsonl(&path).await
            }
            _ => Ok(Vec::new()),
        }
    }

    /// Append one JSON-line record to `base_path/file_name`, fsync-ing before
    /// returning so a successful result is a durability guarantee.
    async fn append_jsonl(&self, base_path: &str, file_name: &str, line: String) -> Result<()> {
        let path = Path::new(base_path).join(file_name);
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
            .map_err(|e| {
                anyhow::anyhow!("Failed to open persistence file '{}': {e}", path.display())
            })?;
        file.write_all(line.as_bytes()).await.map_err(|e| {
            anyhow::anyhow!(
                "Failed to write to persistence file '{}': {e}",
                path.display()
            )
        })?;
        file.write_all(b"\n").await.map_err(|e| {
            anyhow::anyhow!(
                "Failed to write to persistence file '{}': {e}",
                path.display()
            )
        })?;
        file.flush()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to flush '{}': {e}", path.display()))?;
        file.sync_data()
            .await
            .map_err(|e| anyhow::anyhow!("fsync failed for '{}': {e}", path.display()))?;

        self.stats
            .bytes_written
            .fetch_add(line.len() as u64 + 1, Ordering::Relaxed);
        Ok(())
    }

    /// Read and parse a JSON-lines file, skipping (and logging) any corrupt
    /// records instead of failing the whole load.
    async fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
        if !tokio::fs::try_exists(path).await.unwrap_or(false) {
            return Ok(Vec::new());
        }

        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read '{}': {e}", path.display()))?;

        let mut items = Vec::new();
        for (line_no, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<T>(line) {
                Ok(item) => items.push(item),
                Err(e) => warn!(
                    "Skipping corrupt persisted record at {}:{}: {e}",
                    path.display(),
                    line_no + 1
                ),
            }
        }
        Ok(items)
    }

    /// Queue a persistence operation for later batch processing (used for
    /// archival/deletion bookkeeping; regular event/snapshot writes go through
    /// [`Self::persist_event`]/[`Self::persist_snapshot`] synchronously).
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

    /// Execute a single queued persistence operation
    async fn execute_operation(&self, operation: PersistenceOperation) -> Result<()> {
        match &self.backend {
            PersistenceBackend::Memory => Ok(()),
            PersistenceBackend::FileSystem { base_path } => {
                self.execute_filesystem_operation(operation, base_path)
                    .await
            }
            PersistenceBackend::Database { .. } | PersistenceBackend::ObjectStorage { .. } => {
                Err(anyhow::anyhow!(
                    "Persistence backend {:?} is not implemented; use FileSystem or Memory",
                    self.backend
                ))
            }
        }
    }

    /// Execute filesystem persistence operation
    async fn execute_filesystem_operation(
        &self,
        operation: PersistenceOperation,
        base_path: &str,
    ) -> Result<()> {
        match operation {
            PersistenceOperation::StoreEvent(event) => {
                let line = serde_json::to_string(&*event)?;
                self.append_jsonl(base_path, "events.jsonl", line).await?;
            }
            PersistenceOperation::StoreSnapshot(snapshot) => {
                let line = serde_json::to_string(&snapshot)?;
                self.append_jsonl(base_path, "snapshots.jsonl", line)
                    .await?;
            }
            PersistenceOperation::ArchiveEvents(events) => {
                for event in &events {
                    let line = serde_json::to_string(event)?;
                    self.append_jsonl(base_path, "archive.jsonl", line).await?;
                }
            }
            PersistenceOperation::DeleteEvents(sequence_numbers) => {
                let line = serde_json::to_string(&sequence_numbers)?;
                self.append_jsonl(base_path, "deletions.jsonl", line)
                    .await?;
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
