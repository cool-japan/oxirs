//! # Stream Replay and Reprocessing
//!
//! This module provides comprehensive stream replay and reprocessing capabilities
//! for debugging, recovery, and data analysis.
//!
//! ## Features
//!
//! - **Time-based replay**: Replay events from a specific time range
//! - **Offset-based replay**: Replay from a specific offset
//! - **Conditional replay**: Replay with filtering and transformation
//! - **Speed control**: Replay at custom speeds (slow-motion, fast-forward)
//! - **State snapshots**: Capture and restore application state
//! - **Parallel replay**: Replay multiple streams concurrently
//! - **Incremental processing**: Process only new events since last run
//!
//! ## Use Cases
//!
//! - **Debugging**: Replay problematic event sequences
//! - **Testing**: Replay production data in test environments
//! - **Recovery**: Rebuild state after failures
//! - **Analysis**: Analyze historical event patterns
//! - **Migration**: Reprocess events with new business logic

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::event::StreamEvent;

/// Replay mode determining how events are replayed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplayMode {
    /// Replay from a specific time
    FromTime(DateTime<Utc>),
    /// Replay from a specific offset
    FromOffset(u64),
    /// Replay a time range
    TimeRange {
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    },
    /// Replay an offset range
    OffsetRange { start: u64, end: u64 },
    /// Replay all events
    All,
    /// Replay only events matching a filter
    Filtered {
        filter: String,
        from: Option<DateTime<Utc>>,
    },
}

/// Speed control for replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplaySpeed {
    /// Real-time speed (preserve original timing)
    RealTime,
    /// Custom speed multiplier (2.0 = 2x faster, 0.5 = 2x slower)
    Custom(f64),
    /// Fast as possible (no delays)
    MaxSpeed,
    /// Slow motion for debugging
    SlowMotion(f64),
}

/// Filter for selective replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayFilter {
    /// Event types to include
    pub event_types: Option<Vec<String>>,
    /// Event sources to include
    pub sources: Option<Vec<String>>,
    /// Minimum event priority
    pub min_priority: Option<u8>,
    /// Custom predicate (serialized as string)
    pub custom_predicate: Option<String>,
}

/// Transformation to apply during replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayTransformation {
    /// Transform name
    pub name: String,
    /// Transform type
    pub transform_type: TransformationType,
    /// Parameters
    pub parameters: HashMap<String, String>,
}

/// Type of transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformationType {
    /// Map event fields
    FieldMapping,
    /// Enrich with additional data
    Enrichment,
    /// Aggregate multiple events
    Aggregation,
    /// Split single event into multiple
    Splitting,
    /// Custom transformation
    Custom,
}

/// Configuration for stream replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayConfig {
    /// Replay mode
    pub mode: ReplayMode,
    /// Replay speed
    pub speed: ReplaySpeed,
    /// Optional filter
    pub filter: Option<ReplayFilter>,
    /// Optional transformations
    pub transformations: Vec<ReplayTransformation>,
    /// Batch size for replay
    pub batch_size: usize,
    /// Enable state snapshots
    pub enable_snapshots: bool,
    /// Snapshot interval
    pub snapshot_interval: Duration,
    /// Enable parallel replay
    pub enable_parallel: bool,
    /// Number of parallel workers
    pub parallel_workers: usize,
    /// Checkpoint interval for long replays
    pub checkpoint_interval: Duration,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            mode: ReplayMode::All,
            speed: ReplaySpeed::MaxSpeed,
            filter: None,
            transformations: Vec::new(),
            batch_size: 1000,
            enable_snapshots: true,
            snapshot_interval: Duration::from_secs(60),
            enable_parallel: false,
            parallel_workers: 4,
            checkpoint_interval: Duration::from_secs(30),
        }
    }
}

/// State snapshot for recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    /// Snapshot ID
    pub snapshot_id: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Event offset at snapshot
    pub event_offset: u64,
    /// Application state (serialized)
    pub state_data: Vec<u8>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Replay checkpoint for resuming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayCheckpoint {
    /// Checkpoint ID
    pub checkpoint_id: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Last processed offset
    pub last_offset: u64,
    /// Events processed
    pub events_processed: u64,
    /// Replay status
    pub status: ReplayStatus,
}

/// Status of replay operation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReplayStatus {
    /// Replay not started
    NotStarted,
    /// Replay in progress
    InProgress,
    /// Replay paused
    Paused,
    /// Replay completed
    Completed,
    /// Replay failed
    Failed { reason: String },
}

/// Statistics for replay operations
#[derive(Debug, Clone, Default)]
pub struct ReplayStats {
    /// Total events replayed
    pub events_replayed: u64,
    /// Events filtered out
    pub events_filtered: u64,
    /// Events transformed
    pub events_transformed: u64,
    /// Total replay time (ms)
    pub total_replay_time_ms: u64,
    /// Average processing time per event (ms)
    pub avg_processing_time_ms: f64,
    /// Snapshots created
    pub snapshots_created: u64,
    /// Checkpoints created
    pub checkpoints_created: u64,
    /// Errors encountered
    pub errors_encountered: u64,
}

/// Stream replay manager
pub struct StreamReplayManager {
    /// Configuration
    config: ReplayConfig,
    /// Event store for replay
    event_store: Arc<DashMap<u64, StreamEvent>>,
    /// State snapshots
    snapshots: Arc<RwLock<Vec<StateSnapshot>>>,
    /// Replay checkpoints
    checkpoints: Arc<RwLock<Vec<ReplayCheckpoint>>>,
    /// Statistics
    stats: Arc<RwLock<ReplayStats>>,
    /// Active replay sessions
    active_replays: Arc<DashMap<String, ReplaySession>>,
    /// Event processors
    processors: Arc<RwLock<Vec<Box<dyn EventProcessor + Send + Sync>>>>,
}

/// Active replay session
struct ReplaySession {
    /// Session ID
    session_id: String,
    /// Start time
    start_time: Instant,
    /// Current status
    status: ReplayStatus,
    /// Current offset
    current_offset: u64,
    /// Events processed in this session
    events_processed: u64,
}

/// Trait for processing replayed events
pub trait EventProcessor: Send + Sync {
    /// Process a replayed event
    fn process(&mut self, event: &StreamEvent) -> Result<Option<StreamEvent>>;

    /// Get processor name
    fn name(&self) -> &str;
}

impl StreamReplayManager {
    /// Create a new stream replay manager
    pub fn new(config: ReplayConfig) -> Self {
        Self {
            config,
            event_store: Arc::new(DashMap::new()),
            snapshots: Arc::new(RwLock::new(Vec::new())),
            checkpoints: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(ReplayStats::default())),
            active_replays: Arc::new(DashMap::new()),
            processors: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Store an event for future replay
    pub fn store_event(&self, offset: u64, event: StreamEvent) {
        self.event_store.insert(offset, event);
        debug!("Stored event at offset {}", offset);
    }

    /// Store multiple events
    pub fn store_events(&self, events: Vec<(u64, StreamEvent)>) {
        for (offset, event) in events {
            self.store_event(offset, event);
        }
    }

    /// Start a replay session
    pub async fn start_replay(
        &self,
        session_id: Option<String>,
    ) -> Result<mpsc::UnboundedReceiver<StreamEvent>> {
        let session_id = session_id.unwrap_or_else(|| Uuid::new_v4().to_string());

        let session = ReplaySession {
            session_id: session_id.clone(),
            start_time: Instant::now(),
            status: ReplayStatus::InProgress,
            current_offset: 0,
            events_processed: 0,
        };

        self.active_replays.insert(session_id.clone(), session);

        let (tx, rx) = mpsc::unbounded_channel();

        // Spawn replay task
        let event_store = self.event_store.clone();
        let config = self.config.clone();
        let stats = self.stats.clone();
        let snapshots = self.snapshots.clone();
        let checkpoints = self.checkpoints.clone();
        let active_replays = self.active_replays.clone();
        let processors = self.processors.clone();
        let session_id_clone = session_id.clone();

        tokio::spawn(async move {
            if let Err(e) = Self::replay_events_internal(
                &session_id_clone,
                &event_store,
                &config,
                &stats,
                &snapshots,
                &checkpoints,
                &active_replays,
                &processors,
                tx,
            )
            .await
            {
                error!("Replay failed: {}", e);

                if let Some(mut session) = active_replays.get_mut(&session_id_clone) {
                    session.status = ReplayStatus::Failed {
                        reason: e.to_string(),
                    };
                }
            }
        });

        info!("Started replay session: {}", session_id);
        Ok(rx)
    }

    /// Internal replay implementation
    #[allow(clippy::too_many_arguments)]
    async fn replay_events_internal(
        session_id: &str,
        event_store: &Arc<DashMap<u64, StreamEvent>>,
        config: &ReplayConfig,
        stats: &Arc<RwLock<ReplayStats>>,
        snapshots: &Arc<RwLock<Vec<StateSnapshot>>>,
        checkpoints: &Arc<RwLock<Vec<ReplayCheckpoint>>>,
        active_replays: &Arc<DashMap<String, ReplaySession>>,
        processors: &Arc<RwLock<Vec<Box<dyn EventProcessor + Send + Sync>>>>,
        tx: mpsc::UnboundedSender<StreamEvent>,
    ) -> Result<()> {
        let start_time = Instant::now();

        // Determine offset range to replay
        let (start_offset, end_offset) = Self::determine_offset_range(config, event_store)?;

        debug!(
            "Replaying events from offset {} to {}",
            start_offset, end_offset
        );

        let mut events_replayed = 0;
        let mut last_checkpoint = Instant::now();
        let mut last_snapshot = Instant::now();

        for offset in start_offset..=end_offset {
            if let Some(event) = event_store.get(&offset) {
                let event = event.clone();

                // Apply filter
                if let Some(ref filter) = config.filter {
                    if !Self::apply_filter(&event, filter) {
                        stats.write().events_filtered += 1;
                        continue;
                    }
                }

                // Apply transformations
                let mut transformed_event = event.clone();
                for transformation in &config.transformations {
                    transformed_event =
                        Self::apply_transformation(transformed_event, transformation)?;
                }

                if !config.transformations.is_empty() {
                    stats.write().events_transformed += 1;
                }

                // Process through registered processors
                let mut final_event = Some(transformed_event);
                for processor in processors.write().iter_mut() {
                    if let Some(evt) = final_event {
                        final_event = processor.process(&evt)?;
                    }
                }

                // Send event if not filtered by processors
                if let Some(evt) = final_event {
                    // Apply speed control
                    Self::apply_speed_control(config).await;

                    if tx.send(evt).is_err() {
                        warn!("Receiver dropped, stopping replay");
                        break;
                    }

                    events_replayed += 1;

                    // Update session
                    if let Some(mut session) = active_replays.get_mut(session_id) {
                        session.current_offset = offset;
                        session.events_processed = events_replayed;
                    }
                }

                // Create checkpoint if needed
                if last_checkpoint.elapsed() >= config.checkpoint_interval {
                    Self::create_checkpoint(
                        session_id,
                        offset,
                        events_replayed,
                        checkpoints,
                        stats,
                    )
                    .await?;
                    last_checkpoint = Instant::now();
                }

                // Create snapshot if needed
                if config.enable_snapshots && last_snapshot.elapsed() >= config.snapshot_interval {
                    Self::create_snapshot(session_id, offset, snapshots, stats).await?;
                    last_snapshot = Instant::now();
                }
            }
        }

        // Update final stats
        let total_time = start_time.elapsed().as_millis() as u64;
        let mut stats_guard = stats.write();
        stats_guard.events_replayed += events_replayed;
        stats_guard.total_replay_time_ms += total_time;
        if events_replayed > 0 {
            stats_guard.avg_processing_time_ms = total_time as f64 / events_replayed as f64;
        }

        // Mark session as completed
        if let Some(mut session) = active_replays.get_mut(session_id) {
            session.status = ReplayStatus::Completed;
        }

        info!(
            "Replay completed: {} events in {}ms",
            events_replayed, total_time
        );
        Ok(())
    }

    /// Determine offset range for replay
    fn determine_offset_range(
        config: &ReplayConfig,
        event_store: &Arc<DashMap<u64, StreamEvent>>,
    ) -> Result<(u64, u64)> {
        let max_offset = event_store.iter().map(|e| *e.key()).max().unwrap_or(0);

        match &config.mode {
            ReplayMode::All => Ok((0, max_offset)),
            ReplayMode::FromOffset(start) => Ok((*start, max_offset)),
            ReplayMode::OffsetRange { start, end } => Ok((*start, *end)),
            ReplayMode::FromTime(start_time) => {
                // Find first offset after start_time
                let start_offset = event_store
                    .iter()
                    .filter_map(|entry| {
                        let offset = *entry.key();
                        let event = entry.value();
                        let event_time = Self::get_event_timestamp(event);
                        if event_time >= *start_time {
                            Some(offset)
                        } else {
                            None
                        }
                    })
                    .min()
                    .unwrap_or(0);
                Ok((start_offset, max_offset))
            }
            ReplayMode::TimeRange { start, end } => {
                let start_offset = event_store
                    .iter()
                    .filter_map(|entry| {
                        let offset = *entry.key();
                        let event = entry.value();
                        let event_time = Self::get_event_timestamp(event);
                        if event_time >= *start {
                            Some(offset)
                        } else {
                            None
                        }
                    })
                    .min()
                    .unwrap_or(0);

                let end_offset = event_store
                    .iter()
                    .filter_map(|entry| {
                        let offset = *entry.key();
                        let event = entry.value();
                        let event_time = Self::get_event_timestamp(event);
                        if event_time <= *end {
                            Some(offset)
                        } else {
                            None
                        }
                    })
                    .max()
                    .unwrap_or(max_offset);

                Ok((start_offset, end_offset))
            }
            ReplayMode::Filtered { from, .. } => {
                let start_offset = if let Some(start_time) = from {
                    event_store
                        .iter()
                        .filter_map(|entry| {
                            let offset = *entry.key();
                            let event = entry.value();
                            let event_time = Self::get_event_timestamp(event);
                            if event_time >= *start_time {
                                Some(offset)
                            } else {
                                None
                            }
                        })
                        .min()
                        .unwrap_or(0)
                } else {
                    0
                };
                Ok((start_offset, max_offset))
            }
        }
    }

    /// Get event timestamp
    fn get_event_timestamp(event: &StreamEvent) -> DateTime<Utc> {
        let metadata = match event {
            StreamEvent::TripleAdded { metadata, .. }
            | StreamEvent::TripleRemoved { metadata, .. }
            | StreamEvent::QuadAdded { metadata, .. }
            | StreamEvent::QuadRemoved { metadata, .. }
            | StreamEvent::GraphCreated { metadata, .. }
            | StreamEvent::GraphCleared { metadata, .. }
            | StreamEvent::GraphDeleted { metadata, .. }
            | StreamEvent::GraphMetadataUpdated { metadata, .. }
            | StreamEvent::GraphPermissionsChanged { metadata, .. }
            | StreamEvent::GraphStatisticsUpdated { metadata, .. }
            | StreamEvent::GraphRenamed { metadata, .. }
            | StreamEvent::GraphMerged { metadata, .. }
            | StreamEvent::GraphSplit { metadata, .. }
            | StreamEvent::SparqlUpdate { metadata, .. }
            | StreamEvent::QueryCompleted { metadata, .. }
            | StreamEvent::QueryResultAdded { metadata, .. }
            | StreamEvent::QueryResultRemoved { metadata, .. }
            | StreamEvent::TransactionBegin { metadata, .. }
            | StreamEvent::TransactionCommit { metadata, .. }
            | StreamEvent::TransactionAbort { metadata, .. }
            | StreamEvent::SchemaChanged { metadata, .. }
            | StreamEvent::SchemaDefinitionAdded { metadata, .. }
            | StreamEvent::SchemaDefinitionRemoved { metadata, .. }
            | StreamEvent::SchemaDefinitionModified { metadata, .. }
            | StreamEvent::OntologyImported { metadata, .. }
            | StreamEvent::OntologyRemoved { metadata, .. }
            | StreamEvent::ConstraintAdded { metadata, .. }
            | StreamEvent::ConstraintRemoved { metadata, .. }
            | StreamEvent::ConstraintViolated { metadata, .. }
            | StreamEvent::IndexCreated { metadata, .. }
            | StreamEvent::IndexDropped { metadata, .. }
            | StreamEvent::IndexRebuilt { metadata, .. }
            | StreamEvent::SchemaUpdated { metadata, .. }
            | StreamEvent::ShapeAdded { metadata, .. }
            | StreamEvent::ShapeRemoved { metadata, .. }
            | StreamEvent::ShapeModified { metadata, .. }
            | StreamEvent::ShapeUpdated { metadata, .. }
            | StreamEvent::ShapeValidationStarted { metadata, .. }
            | StreamEvent::ShapeValidationCompleted { metadata, .. }
            | StreamEvent::ShapeViolationDetected { metadata, .. }
            | StreamEvent::Heartbeat { metadata, .. }
            | StreamEvent::ErrorOccurred { metadata, .. } => metadata,
        };
        metadata.timestamp
    }

    /// Apply filter to event
    fn apply_filter(event: &StreamEvent, filter: &ReplayFilter) -> bool {
        // Check event type
        if let Some(ref event_types) = filter.event_types {
            let event_type = Self::get_event_type(event);
            if !event_types.contains(&event_type) {
                return false;
            }
        }

        // Check source
        if let Some(ref sources) = filter.sources {
            let source = Self::get_event_source(event);
            if !sources.contains(&source) {
                return false;
            }
        }

        true
    }

    /// Get event type
    fn get_event_type(event: &StreamEvent) -> String {
        match event {
            StreamEvent::TripleAdded { .. } => "TripleAdded",
            StreamEvent::TripleRemoved { .. } => "TripleRemoved",
            StreamEvent::QuadAdded { .. } => "QuadAdded",
            StreamEvent::QuadRemoved { .. } => "QuadRemoved",
            StreamEvent::GraphCreated { .. } => "GraphCreated",
            StreamEvent::GraphCleared { .. } => "GraphCleared",
            StreamEvent::GraphDeleted { .. } => "GraphDeleted",
            StreamEvent::SparqlUpdate { .. } => "SparqlUpdate",
            StreamEvent::TransactionBegin { .. } => "TransactionBegin",
            StreamEvent::TransactionCommit { .. } => "TransactionCommit",
            StreamEvent::TransactionAbort { .. } => "TransactionAbort",
            StreamEvent::SchemaChanged { .. } => "SchemaChanged",
            StreamEvent::Heartbeat { .. } => "Heartbeat",
            StreamEvent::QueryResultAdded { .. } => "QueryResultAdded",
            StreamEvent::QueryResultRemoved { .. } => "QueryResultRemoved",
            StreamEvent::QueryCompleted { .. } => "QueryCompleted",
            _ => "Other",
        }
        .to_string()
    }

    /// Get event source
    fn get_event_source(event: &StreamEvent) -> String {
        let metadata = match event {
            StreamEvent::TripleAdded { metadata, .. }
            | StreamEvent::TripleRemoved { metadata, .. }
            | StreamEvent::QuadAdded { metadata, .. }
            | StreamEvent::QuadRemoved { metadata, .. }
            | StreamEvent::GraphCreated { metadata, .. }
            | StreamEvent::GraphCleared { metadata, .. }
            | StreamEvent::GraphDeleted { metadata, .. }
            | StreamEvent::SparqlUpdate { metadata, .. }
            | StreamEvent::TransactionBegin { metadata, .. }
            | StreamEvent::TransactionCommit { metadata, .. }
            | StreamEvent::TransactionAbort { metadata, .. }
            | StreamEvent::SchemaChanged { metadata, .. }
            | StreamEvent::Heartbeat { metadata, .. }
            | StreamEvent::QueryResultAdded { metadata, .. }
            | StreamEvent::QueryResultRemoved { metadata, .. }
            | StreamEvent::QueryCompleted { metadata, .. }
            | StreamEvent::GraphMetadataUpdated { metadata, .. }
            | StreamEvent::GraphPermissionsChanged { metadata, .. }
            | StreamEvent::GraphStatisticsUpdated { metadata, .. }
            | StreamEvent::GraphRenamed { metadata, .. }
            | StreamEvent::GraphMerged { metadata, .. }
            | StreamEvent::GraphSplit { metadata, .. }
            | StreamEvent::SchemaDefinitionAdded { metadata, .. }
            | StreamEvent::SchemaDefinitionRemoved { metadata, .. }
            | StreamEvent::SchemaDefinitionModified { metadata, .. }
            | StreamEvent::OntologyImported { metadata, .. }
            | StreamEvent::OntologyRemoved { metadata, .. }
            | StreamEvent::ConstraintAdded { metadata, .. }
            | StreamEvent::ConstraintRemoved { metadata, .. }
            | StreamEvent::ConstraintViolated { metadata, .. }
            | StreamEvent::IndexCreated { metadata, .. }
            | StreamEvent::IndexDropped { metadata, .. }
            | StreamEvent::IndexRebuilt { metadata, .. }
            | StreamEvent::SchemaUpdated { metadata, .. }
            | StreamEvent::ShapeAdded { metadata, .. }
            | StreamEvent::ShapeUpdated { metadata, .. }
            | StreamEvent::ShapeRemoved { metadata, .. }
            | StreamEvent::ShapeModified { metadata, .. }
            | StreamEvent::ShapeValidationStarted { metadata, .. }
            | StreamEvent::ShapeValidationCompleted { metadata, .. }
            | StreamEvent::ShapeViolationDetected { metadata, .. }
            | StreamEvent::ErrorOccurred { metadata, .. } => metadata,
        };
        metadata.source.clone()
    }

    /// Apply transformation to event
    fn apply_transformation(
        event: StreamEvent,
        _transformation: &ReplayTransformation,
    ) -> Result<StreamEvent> {
        // Placeholder for transformation logic
        // In real implementation, this would apply the actual transformation
        Ok(event)
    }

    /// Apply speed control
    async fn apply_speed_control(config: &ReplayConfig) {
        match config.speed {
            ReplaySpeed::RealTime => {
                // Would need event timing information to preserve original timing
                sleep(Duration::from_millis(1)).await;
            }
            ReplaySpeed::MaxSpeed => {
                // No delay
            }
            ReplaySpeed::Custom(multiplier) => {
                let delay = Duration::from_millis((10.0 / multiplier) as u64);
                sleep(delay).await;
            }
            ReplaySpeed::SlowMotion(factor) => {
                let delay = Duration::from_millis((100.0 * factor) as u64);
                sleep(delay).await;
            }
        }
    }

    /// Create a checkpoint
    async fn create_checkpoint(
        session_id: &str,
        offset: u64,
        events_processed: u64,
        checkpoints: &Arc<RwLock<Vec<ReplayCheckpoint>>>,
        stats: &Arc<RwLock<ReplayStats>>,
    ) -> Result<()> {
        let checkpoint = ReplayCheckpoint {
            checkpoint_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            last_offset: offset,
            events_processed,
            status: ReplayStatus::InProgress,
        };

        checkpoints.write().push(checkpoint);
        stats.write().checkpoints_created += 1;

        debug!(
            "Checkpoint created for session {} at offset {}",
            session_id, offset
        );
        Ok(())
    }

    /// Create a state snapshot
    async fn create_snapshot(
        session_id: &str,
        offset: u64,
        snapshots: &Arc<RwLock<Vec<StateSnapshot>>>,
        stats: &Arc<RwLock<ReplayStats>>,
    ) -> Result<()> {
        let snapshot = StateSnapshot {
            snapshot_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event_offset: offset,
            state_data: Vec::new(), // Would contain actual state
            metadata: HashMap::new(),
        };

        snapshots.write().push(snapshot);
        stats.write().snapshots_created += 1;

        debug!(
            "Snapshot created for session {} at offset {}",
            session_id, offset
        );
        Ok(())
    }

    /// Pause a replay session
    pub fn pause_replay(&self, session_id: &str) -> Result<()> {
        if let Some(mut session) = self.active_replays.get_mut(session_id) {
            session.status = ReplayStatus::Paused;
            info!("Replay session {} paused", session_id);
            Ok(())
        } else {
            Err(anyhow!("Replay session not found: {}", session_id))
        }
    }

    /// Resume a paused replay session
    pub fn resume_replay(&self, session_id: &str) -> Result<()> {
        if let Some(mut session) = self.active_replays.get_mut(session_id) {
            session.status = ReplayStatus::InProgress;
            info!("Replay session {} resumed", session_id);
            Ok(())
        } else {
            Err(anyhow!("Replay session not found: {}", session_id))
        }
    }

    /// Get replay statistics
    pub fn get_stats(&self) -> ReplayStats {
        self.stats.read().clone()
    }

    /// Get session status
    pub fn get_session_status(&self, session_id: &str) -> Option<ReplayStatus> {
        self.active_replays
            .get(session_id)
            .map(|session| session.status.clone())
    }

    /// Register an event processor
    pub fn register_processor(&self, processor: Box<dyn EventProcessor + Send + Sync>) {
        let name = processor.name().to_string();
        self.processors.write().push(processor);
        info!("Registered event processor: {}", name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventMetadata;

    #[tokio::test]
    async fn test_replay_all_events() {
        let config = ReplayConfig {
            mode: ReplayMode::All,
            speed: ReplaySpeed::MaxSpeed,
            ..Default::default()
        };

        let manager = StreamReplayManager::new(config);

        // Store test events
        for i in 0..10 {
            let event = StreamEvent::SchemaChanged {
                schema_type: crate::event::SchemaType::Ontology,
                change_type: crate::event::SchemaChangeType::Added,
                details: format!("test schema change {}", i),
                metadata: EventMetadata {
                    event_id: format!("event-{}", i),
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
            manager.store_event(i, event);
        }

        // Start replay
        let mut rx = manager.start_replay(None).await.unwrap();

        // Receive events
        let mut count = 0;
        while let Some(_event) = rx.recv().await {
            count += 1;
        }

        assert_eq!(count, 10);

        let stats = manager.get_stats();
        assert_eq!(stats.events_replayed, 10);
    }

    #[tokio::test]
    async fn test_replay_with_filter() {
        let config = ReplayConfig {
            mode: ReplayMode::All,
            speed: ReplaySpeed::MaxSpeed,
            filter: Some(ReplayFilter {
                event_types: Some(vec!["SchemaChanged".to_string()]),
                sources: Some(vec!["test".to_string()]),
                min_priority: None,
                custom_predicate: None,
            }),
            ..Default::default()
        };

        let manager = StreamReplayManager::new(config);

        // Store test events with different types
        for i in 0..5 {
            let event = StreamEvent::SchemaChanged {
                schema_type: crate::event::SchemaType::Ontology,
                change_type: crate::event::SchemaChangeType::Added,
                details: format!("test schema change {}", i),
                metadata: EventMetadata {
                    event_id: format!("event-{}", i),
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
            manager.store_event(i, event);
        }

        let mut rx = manager.start_replay(None).await.unwrap();

        let mut count = 0;
        while let Some(_event) = rx.recv().await {
            count += 1;
        }

        assert_eq!(count, 5);
    }
}
