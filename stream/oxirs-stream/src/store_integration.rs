//! # OxiRS Store Integration
//!
//! This module provides deep integration between oxirs-stream and oxirs-core store
//! with change detection, real-time updates, and bi-directional synchronization.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, RwLock};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::{
    EventMetadata, StreamEvent, StreamProducer, StreamConsumer,
    StreamConfig, patch::RdfPatch, PatchOperation,
};

/// Store change detector for monitoring RDF store changes
pub struct StoreChangeDetector {
    /// Store connection
    store: Arc<dyn RdfStore>,
    /// Change detection strategy
    strategy: ChangeDetectionStrategy,
    /// Stream producer for publishing changes
    producer: Arc<RwLock<StreamProducer>>,
    /// Change buffer for batching
    change_buffer: Arc<RwLock<Vec<StoreChange>>>,
    /// Configuration
    config: ChangeDetectorConfig,
    /// Statistics
    stats: Arc<RwLock<ChangeDetectorStats>>,
    /// Change event notifier
    change_notifier: broadcast::Sender<StoreChangeEvent>,
}

/// RDF Store trait for abstraction
#[async_trait::async_trait]
pub trait RdfStore: Send + Sync {
    /// Get current transaction log position
    async fn get_transaction_log_position(&self) -> Result<u64>;
    
    /// Read transaction log from position
    async fn read_transaction_log(&self, from: u64, limit: usize) -> Result<Vec<TransactionLogEntry>>;
    
    /// Subscribe to store changes
    async fn subscribe_changes(&self) -> Result<mpsc::Receiver<StoreChange>>;
    
    /// Get store statistics
    async fn get_statistics(&self) -> Result<StoreStatistics>;
    
    /// Apply a patch to the store
    async fn apply_patch(&self, patch: &RdfPatch) -> Result<()>;
    
    /// Execute SPARQL update
    async fn execute_update(&self, update: &str) -> Result<()>;
    
    /// Query store
    async fn query(&self, query: &str) -> Result<QueryResult>;
}

/// Change detection strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeDetectionStrategy {
    /// Transaction log tailing
    TransactionLog {
        poll_interval: Duration,
        batch_size: usize,
    },
    /// Trigger-based detection
    TriggerBased {
        trigger_types: Vec<TriggerType>,
    },
    /// Polling-based detection
    Polling {
        poll_interval: Duration,
        snapshot_interval: Duration,
    },
    /// Event sourcing
    EventSourcing {
        event_store_url: String,
    },
    /// Hybrid approach
    Hybrid {
        primary: Box<ChangeDetectionStrategy>,
        fallback: Box<ChangeDetectionStrategy>,
    },
}

/// Trigger types for trigger-based detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerType {
    Insert,
    Delete,
    Update,
    GraphChange,
    SchemaChange,
}

/// Store change event
#[derive(Debug, Clone)]
pub struct StoreChange {
    pub change_type: ChangeType,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub transaction_id: Option<String>,
    pub user: Option<String>,
    pub affected_triples: Vec<Triple>,
    pub metadata: HashMap<String, String>,
}

/// Change types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    TripleAdded,
    TripleRemoved,
    GraphCreated,
    GraphDeleted,
    GraphCleared,
    BulkUpdate,
    SchemaUpdate,
}

/// Triple representation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub graph: Option<String>,
}

/// Transaction log entry
#[derive(Debug, Clone)]
pub struct TransactionLogEntry {
    pub position: u64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub transaction_id: String,
    pub operations: Vec<LogOperation>,
    pub metadata: HashMap<String, String>,
}

/// Log operation types
#[derive(Debug, Clone)]
pub enum LogOperation {
    Add(Triple),
    Remove(Triple),
    CreateGraph(String),
    DeleteGraph(String),
    ClearGraph(Option<String>),
}

/// Store statistics
#[derive(Debug, Clone)]
pub struct StoreStatistics {
    pub total_triples: u64,
    pub total_graphs: u64,
    pub transaction_count: u64,
    pub last_modified: chrono::DateTime<chrono::Utc>,
}

/// Query result placeholder
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub bindings: Vec<HashMap<String, String>>,
}

/// Change detector configuration
#[derive(Debug, Clone)]
pub struct ChangeDetectorConfig {
    /// Maximum buffer size before flushing
    pub buffer_size: usize,
    /// Buffer flush interval
    pub flush_interval: Duration,
    /// Enable deduplication
    pub enable_deduplication: bool,
    /// Deduplication window
    pub dedup_window: Duration,
    /// Enable change compression
    pub enable_compression: bool,
    /// Minimum batch size for compression
    pub compression_threshold: usize,
}

impl Default for ChangeDetectorConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            flush_interval: Duration::from_millis(100),
            enable_deduplication: true,
            dedup_window: Duration::from_secs(60),
            enable_compression: true,
            compression_threshold: 100,
        }
    }
}

/// Change detector statistics
#[derive(Debug, Default, Clone)]
pub struct ChangeDetectorStats {
    pub changes_detected: u64,
    pub changes_published: u64,
    pub changes_deduplicated: u64,
    pub batches_compressed: u64,
    pub errors: u64,
    pub last_position: u64,
    pub lag_ms: u64,
}

/// Store change event notifications
#[derive(Debug, Clone)]
pub enum StoreChangeEvent {
    /// Changes detected
    ChangesDetected { count: usize },
    /// Changes published
    ChangesPublished { count: usize },
    /// Error occurred
    Error { message: String },
    /// Lag detected
    LagDetected { lag_ms: u64 },
}

impl StoreChangeDetector {
    /// Create a new store change detector
    pub async fn new(
        store: Arc<dyn RdfStore>,
        strategy: ChangeDetectionStrategy,
        producer: Arc<RwLock<StreamProducer>>,
        config: ChangeDetectorConfig,
    ) -> Result<Self> {
        let (tx, _) = broadcast::channel(1000);
        
        Ok(Self {
            store,
            strategy,
            producer,
            change_buffer: Arc::new(RwLock::new(Vec::new())),
            config,
            stats: Arc::new(RwLock::new(ChangeDetectorStats::default())),
            change_notifier: tx,
        })
    }

    /// Start change detection
    pub async fn start(&self) -> Result<()> {
        match &self.strategy {
            ChangeDetectionStrategy::TransactionLog { poll_interval, batch_size } => {
                self.start_transaction_log_tailing(*poll_interval, *batch_size).await
            }
            ChangeDetectionStrategy::TriggerBased { .. } => {
                self.start_trigger_based_detection().await
            }
            ChangeDetectionStrategy::Polling { poll_interval, .. } => {
                self.start_polling_detection(*poll_interval).await
            }
            ChangeDetectionStrategy::EventSourcing { .. } => {
                self.start_event_sourcing().await
            }
            ChangeDetectionStrategy::Hybrid { primary, fallback } => {
                self.start_hybrid_detection().await
            }
        }
    }

    /// Start transaction log tailing
    async fn start_transaction_log_tailing(&self, poll_interval: Duration, batch_size: usize) -> Result<()> {
        let store = self.store.clone();
        let buffer = self.change_buffer.clone();
        let stats = self.stats.clone();
        let notifier = self.change_notifier.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(poll_interval);
            let mut last_position = 0u64;
            
            loop {
                interval.tick().await;
                
                match store.read_transaction_log(last_position, batch_size).await {
                    Ok(entries) => {
                        if entries.is_empty() {
                            continue;
                        }
                        
                        let mut changes = Vec::new();
                        
                        for entry in entries {
                            last_position = last_position.max(entry.position + 1);
                            
                            for op in entry.operations {
                                let change = match op {
                                    LogOperation::Add(triple) => StoreChange {
                                        change_type: ChangeType::TripleAdded,
                                        timestamp: entry.timestamp,
                                        transaction_id: Some(entry.transaction_id.clone()),
                                        user: entry.metadata.get("user").cloned(),
                                        affected_triples: vec![triple],
                                        metadata: entry.metadata.clone(),
                                    },
                                    LogOperation::Remove(triple) => StoreChange {
                                        change_type: ChangeType::TripleRemoved,
                                        timestamp: entry.timestamp,
                                        transaction_id: Some(entry.transaction_id.clone()),
                                        user: entry.metadata.get("user").cloned(),
                                        affected_triples: vec![triple],
                                        metadata: entry.metadata.clone(),
                                    },
                                    LogOperation::CreateGraph(graph) => StoreChange {
                                        change_type: ChangeType::GraphCreated,
                                        timestamp: entry.timestamp,
                                        transaction_id: Some(entry.transaction_id.clone()),
                                        user: entry.metadata.get("user").cloned(),
                                        affected_triples: vec![],
                                        metadata: {
                                            let mut meta = entry.metadata.clone();
                                            meta.insert("graph".to_string(), graph);
                                            meta
                                        },
                                    },
                                    LogOperation::DeleteGraph(graph) => StoreChange {
                                        change_type: ChangeType::GraphDeleted,
                                        timestamp: entry.timestamp,
                                        transaction_id: Some(entry.transaction_id.clone()),
                                        user: entry.metadata.get("user").cloned(),
                                        affected_triples: vec![],
                                        metadata: {
                                            let mut meta = entry.metadata.clone();
                                            meta.insert("graph".to_string(), graph);
                                            meta
                                        },
                                    },
                                    LogOperation::ClearGraph(graph) => StoreChange {
                                        change_type: ChangeType::GraphCleared,
                                        timestamp: entry.timestamp,
                                        transaction_id: Some(entry.transaction_id.clone()),
                                        user: entry.metadata.get("user").cloned(),
                                        affected_triples: vec![],
                                        metadata: {
                                            let mut meta = entry.metadata.clone();
                                            if let Some(g) = graph {
                                                meta.insert("graph".to_string(), g);
                                            }
                                            meta
                                        },
                                    },
                                };
                                
                                changes.push(change);
                            }
                        }
                        
                        // Update statistics
                        {
                            let mut stats_guard = stats.write().await;
                            stats_guard.changes_detected += changes.len() as u64;
                            stats_guard.last_position = last_position;
                        }
                        
                        // Add to buffer
                        buffer.write().await.extend(changes.clone());
                        
                        // Notify
                        let _ = notifier.send(StoreChangeEvent::ChangesDetected {
                            count: changes.len(),
                        });
                        
                        debug!("Detected {} changes from transaction log", changes.len());
                    }
                    Err(e) => {
                        error!("Failed to read transaction log: {}", e);
                        stats.write().await.errors += 1;
                        let _ = notifier.send(StoreChangeEvent::Error {
                            message: e.to_string(),
                        });
                    }
                }
            }
        });
        
        // Start buffer flusher
        self.start_buffer_flusher().await;
        
        Ok(())
    }

    /// Start trigger-based detection
    async fn start_trigger_based_detection(&self) -> Result<()> {
        let mut receiver = self.store.subscribe_changes().await?;
        let buffer = self.change_buffer.clone();
        let stats = self.stats.clone();
        let notifier = self.change_notifier.clone();
        
        tokio::spawn(async move {
            while let Some(change) = receiver.recv().await {
                // Update statistics
                stats.write().await.changes_detected += 1;
                
                // Add to buffer
                buffer.write().await.push(change);
                
                // Notify
                let _ = notifier.send(StoreChangeEvent::ChangesDetected { count: 1 });
            }
        });
        
        // Start buffer flusher
        self.start_buffer_flusher().await;
        
        Ok(())
    }

    /// Start polling-based detection
    async fn start_polling_detection(&self, poll_interval: Duration) -> Result<()> {
        let store = self.store.clone();
        let buffer = self.change_buffer.clone();
        let stats = self.stats.clone();
        
        // This would implement snapshot-based change detection
        // by comparing periodic snapshots of the store
        
        tokio::spawn(async move {
            let mut interval = interval(poll_interval);
            let mut last_snapshot: Option<StoreSnapshot> = None;
            
            loop {
                interval.tick().await;
                
                // Get current snapshot
                match Self::take_snapshot(&store).await {
                    Ok(current_snapshot) => {
                        if let Some(last) = &last_snapshot {
                            // Compare snapshots and detect changes
                            let changes = Self::compare_snapshots(last, &current_snapshot);
                            
                            if !changes.is_empty() {
                                stats.write().await.changes_detected += changes.len() as u64;
                                buffer.write().await.extend(changes);
                            }
                        }
                        
                        last_snapshot = Some(current_snapshot);
                    }
                    Err(e) => {
                        error!("Failed to take snapshot: {}", e);
                        stats.write().await.errors += 1;
                    }
                }
            }
        });
        
        Ok(())
    }

    /// Start event sourcing
    async fn start_event_sourcing(&self) -> Result<()> {
        // This would connect to an external event store
        // and consume events from it
        warn!("Event sourcing not implemented yet");
        Ok(())
    }

    /// Start hybrid detection
    async fn start_hybrid_detection(&self) -> Result<()> {
        // This would start both primary and fallback strategies
        // and coordinate between them
        warn!("Hybrid detection not implemented yet");
        Ok(())
    }

    /// Start buffer flusher
    async fn start_buffer_flusher(&self) {
        let buffer = self.change_buffer.clone();
        let producer = self.producer.clone();
        let config = self.config.clone();
        let stats = self.stats.clone();
        let notifier = self.change_notifier.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(config.flush_interval);
            let mut dedup_cache = if config.enable_deduplication {
                Some(DedupCache::new(config.dedup_window))
            } else {
                None
            };
            
            loop {
                interval.tick().await;
                
                let mut changes = {
                    let mut buffer_guard = buffer.write().await;
                    std::mem::take(&mut *buffer_guard)
                };
                
                if changes.is_empty() {
                    continue;
                }
                
                // Apply deduplication if enabled
                if let Some(cache) = &mut dedup_cache {
                    let original_count = changes.len();
                    changes = cache.deduplicate(changes);
                    let dedup_count = original_count - changes.len();
                    
                    if dedup_count > 0 {
                        stats.write().await.changes_deduplicated += dedup_count as u64;
                        debug!("Deduplicated {} changes", dedup_count);
                    }
                }
                
                // Convert to stream events
                let events = Self::convert_to_stream_events(changes);
                
                // Publish events
                match producer.write().await.publish_batch(events).await {
                    Ok(_) => {
                        let count = events.len();
                        stats.write().await.changes_published += count as u64;
                        let _ = notifier.send(StoreChangeEvent::ChangesPublished { count });
                        debug!("Published {} changes to stream", count);
                    }
                    Err(e) => {
                        error!("Failed to publish changes: {}", e);
                        stats.write().await.errors += 1;
                        let _ = notifier.send(StoreChangeEvent::Error {
                            message: e.to_string(),
                        });
                    }
                }
            }
        });
    }

    /// Convert store changes to stream events
    fn convert_to_stream_events(changes: Vec<StoreChange>) -> Vec<StreamEvent> {
        changes
            .into_iter()
            .flat_map(|change| {
                let metadata = EventMetadata {
                    event_id: uuid::Uuid::new_v4().to_string(),
                    timestamp: change.timestamp,
                    source: "store-change-detector".to_string(),
                    user: change.user,
                    context: change.transaction_id.clone(),
                    caused_by: None,
                    version: "1.0".to_string(),
                    properties: change.metadata,
                    checksum: None,
                };
                
                match change.change_type {
                    ChangeType::TripleAdded => {
                        change.affected_triples.into_iter().map(|triple| {
                            StreamEvent::TripleAdded {
                                subject: triple.subject,
                                predicate: triple.predicate,
                                object: triple.object,
                                graph: triple.graph,
                                metadata: metadata.clone(),
                            }
                        }).collect()
                    }
                    ChangeType::TripleRemoved => {
                        change.affected_triples.into_iter().map(|triple| {
                            StreamEvent::TripleRemoved {
                                subject: triple.subject,
                                predicate: triple.predicate,
                                object: triple.object,
                                graph: triple.graph,
                                metadata: metadata.clone(),
                            }
                        }).collect()
                    }
                    ChangeType::GraphCreated => {
                        if let Some(graph) = change.metadata.get("graph") {
                            vec![StreamEvent::GraphCreated {
                                graph: graph.clone(),
                                metadata,
                            }]
                        } else {
                            vec![]
                        }
                    }
                    ChangeType::GraphDeleted => {
                        if let Some(graph) = change.metadata.get("graph") {
                            vec![StreamEvent::GraphDeleted {
                                graph: graph.clone(),
                                metadata,
                            }]
                        } else {
                            vec![]
                        }
                    }
                    ChangeType::GraphCleared => {
                        vec![StreamEvent::GraphCleared {
                            graph: change.metadata.get("graph").cloned(),
                            metadata,
                        }]
                    }
                    _ => vec![],
                }
            })
            .collect()
    }

    /// Take a snapshot of the store
    async fn take_snapshot(store: &Arc<dyn RdfStore>) -> Result<StoreSnapshot> {
        // This would query the store to get current state
        // For now, return a placeholder
        Ok(StoreSnapshot {
            timestamp: chrono::Utc::now(),
            triple_count: 0,
            graph_count: 0,
            checksum: String::new(),
        })
    }

    /// Compare two snapshots to detect changes
    fn compare_snapshots(old: &StoreSnapshot, new: &StoreSnapshot) -> Vec<StoreChange> {
        // This would implement snapshot comparison logic
        vec![]
    }

    /// Get statistics
    pub async fn get_stats(&self) -> ChangeDetectorStats {
        self.stats.read().await.clone()
    }

    /// Subscribe to change events
    pub fn subscribe(&self) -> broadcast::Receiver<StoreChangeEvent> {
        self.change_notifier.subscribe()
    }
}

/// Store snapshot for polling-based detection
#[derive(Debug, Clone)]
struct StoreSnapshot {
    timestamp: chrono::DateTime<chrono::Utc>,
    triple_count: u64,
    graph_count: u64,
    checksum: String,
}

/// Deduplication cache
struct DedupCache {
    seen: Arc<RwLock<HashSet<String>>>,
    window: Duration,
    last_cleanup: Arc<RwLock<Instant>>,
}

impl DedupCache {
    fn new(window: Duration) -> Self {
        Self {
            seen: Arc::new(RwLock::new(HashSet::new())),
            window,
            last_cleanup: Arc::new(RwLock::new(Instant::now())),
        }
    }
    
    fn deduplicate(&mut self, changes: Vec<StoreChange>) -> Vec<StoreChange> {
        // This would implement proper deduplication logic
        // For now, return all changes
        changes
    }
}

/// Real-time update manager for pushing updates to subscribers
pub struct RealtimeUpdateManager {
    /// Update subscribers
    subscribers: Arc<RwLock<HashMap<String, UpdateSubscriber>>>,
    /// Stream consumer
    consumer: Arc<RwLock<StreamConsumer>>,
    /// Update filters
    filters: Arc<RwLock<Vec<UpdateFilter>>>,
    /// Configuration
    config: UpdateManagerConfig,
    /// Statistics
    stats: Arc<RwLock<UpdateManagerStats>>,
}

/// Update subscriber
#[derive(Debug)]
struct UpdateSubscriber {
    id: String,
    channel: UpdateChannel,
    filters: Vec<UpdateFilter>,
    created_at: Instant,
    last_update: Option<Instant>,
    update_count: u64,
}

/// Update channel types
#[derive(Debug)]
enum UpdateChannel {
    /// WebSocket connection
    WebSocket(mpsc::Sender<UpdateNotification>),
    /// Server-sent events
    ServerSentEvents(mpsc::Sender<UpdateNotification>),
    /// Webhook
    Webhook { url: String, headers: HashMap<String, String> },
    /// Message queue
    MessageQueue { topic: String },
}

/// Update filter
#[derive(Debug, Clone)]
pub struct UpdateFilter {
    /// Filter by graph
    pub graph_filter: Option<String>,
    /// Filter by subject pattern
    pub subject_pattern: Option<regex::Regex>,
    /// Filter by predicate
    pub predicate_filter: Option<String>,
    /// Filter by change type
    pub change_types: Option<Vec<ChangeType>>,
}

/// Update notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateNotification {
    pub id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub changes: Vec<ChangeNotification>,
    pub metadata: HashMap<String, String>,
}

/// Change notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeNotification {
    pub change_type: String,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub graph: Option<String>,
}

/// Update manager configuration
#[derive(Debug, Clone)]
pub struct UpdateManagerConfig {
    /// Maximum subscribers
    pub max_subscribers: usize,
    /// Update batch size
    pub batch_size: usize,
    /// Batch timeout
    pub batch_timeout: Duration,
    /// Enable filtering
    pub enable_filtering: bool,
    /// Retry failed webhooks
    pub retry_webhooks: bool,
    /// Webhook timeout
    pub webhook_timeout: Duration,
}

impl Default for UpdateManagerConfig {
    fn default() -> Self {
        Self {
            max_subscribers: 1000,
            batch_size: 100,
            batch_timeout: Duration::from_millis(50),
            enable_filtering: true,
            retry_webhooks: true,
            webhook_timeout: Duration::from_secs(5),
        }
    }
}

/// Update manager statistics
#[derive(Debug, Default, Clone)]
pub struct UpdateManagerStats {
    pub total_subscribers: usize,
    pub active_subscribers: usize,
    pub updates_sent: u64,
    pub updates_filtered: u64,
    pub webhook_failures: u64,
    pub avg_latency_ms: f64,
}

impl RealtimeUpdateManager {
    /// Create a new realtime update manager
    pub async fn new(
        consumer: Arc<RwLock<StreamConsumer>>,
        config: UpdateManagerConfig,
    ) -> Result<Self> {
        Ok(Self {
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            consumer,
            filters: Arc::new(RwLock::new(Vec::new())),
            config,
            stats: Arc::new(RwLock::new(UpdateManagerStats::default())),
        })
    }
    
    /// Start processing updates
    pub async fn start(&self) -> Result<()> {
        let consumer = self.consumer.clone();
        let subscribers = self.subscribers.clone();
        let filters = self.filters.clone();
        let config = self.config.clone();
        let stats = self.stats.clone();
        
        tokio::spawn(async move {
            let mut batch = Vec::new();
            let mut batch_timer = Instant::now();
            
            loop {
                // Try to consume an event
                match tokio::time::timeout(
                    Duration::from_millis(10),
                    consumer.write().await.consume()
                ).await {
                    Ok(Ok(Some(event))) => {
                        batch.push(event);
                        
                        // Check if batch is ready
                        if batch.len() >= config.batch_size || 
                           batch_timer.elapsed() > config.batch_timeout {
                            Self::process_batch(
                                batch.drain(..).collect(),
                                &subscribers,
                                &filters,
                                &config,
                                &stats,
                            ).await;
                            batch_timer = Instant::now();
                        }
                    }
                    Ok(Ok(None)) => {
                        // No event available
                        if !batch.is_empty() && batch_timer.elapsed() > config.batch_timeout {
                            Self::process_batch(
                                batch.drain(..).collect(),
                                &subscribers,
                                &filters,
                                &config,
                                &stats,
                            ).await;
                            batch_timer = Instant::now();
                        }
                    }
                    Ok(Err(e)) => {
                        error!("Consumer error: {}", e);
                    }
                    Err(_) => {
                        // Timeout - check batch
                        if !batch.is_empty() && batch_timer.elapsed() > config.batch_timeout {
                            Self::process_batch(
                                batch.drain(..).collect(),
                                &subscribers,
                                &filters,
                                &config,
                                &stats,
                            ).await;
                            batch_timer = Instant::now();
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Process a batch of events
    async fn process_batch(
        events: Vec<StreamEvent>,
        subscribers: &Arc<RwLock<HashMap<String, UpdateSubscriber>>>,
        filters: &Arc<RwLock<Vec<UpdateFilter>>>,
        config: &UpdateManagerConfig,
        stats: &Arc<RwLock<UpdateManagerStats>>,
    ) {
        let notification = UpdateNotification {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            changes: events.iter().map(Self::event_to_notification).collect(),
            metadata: HashMap::new(),
        };
        
        let subscribers_guard = subscribers.read().await;
        for (id, subscriber) in subscribers_guard.iter() {
            // Apply filters if enabled
            if config.enable_filtering && !subscriber.filters.is_empty() {
                let filtered_changes: Vec<_> = notification.changes.iter()
                    .filter(|change| Self::matches_filters(change, &subscriber.filters))
                    .cloned()
                    .collect();
                
                if filtered_changes.is_empty() {
                    continue;
                }
                
                let filtered_notification = UpdateNotification {
                    changes: filtered_changes,
                    ..notification.clone()
                };
                
                Self::send_to_subscriber(subscriber, filtered_notification, config, stats).await;
            } else {
                Self::send_to_subscriber(subscriber, notification.clone(), config, stats).await;
            }
        }
    }
    
    /// Convert stream event to change notification
    fn event_to_notification(event: &StreamEvent) -> ChangeNotification {
        match event {
            StreamEvent::TripleAdded { subject, predicate, object, graph, .. } => {
                ChangeNotification {
                    change_type: "triple_added".to_string(),
                    subject: Some(subject.clone()),
                    predicate: Some(predicate.clone()),
                    object: Some(object.clone()),
                    graph: graph.clone(),
                }
            }
            StreamEvent::TripleRemoved { subject, predicate, object, graph, .. } => {
                ChangeNotification {
                    change_type: "triple_removed".to_string(),
                    subject: Some(subject.clone()),
                    predicate: Some(predicate.clone()),
                    object: Some(object.clone()),
                    graph: graph.clone(),
                }
            }
            StreamEvent::GraphCreated { graph, .. } => {
                ChangeNotification {
                    change_type: "graph_created".to_string(),
                    subject: None,
                    predicate: None,
                    object: None,
                    graph: Some(graph.clone()),
                }
            }
            _ => ChangeNotification {
                change_type: "unknown".to_string(),
                subject: None,
                predicate: None,
                object: None,
                graph: None,
            }
        }
    }
    
    /// Check if change matches filters
    fn matches_filters(change: &ChangeNotification, filters: &[UpdateFilter]) -> bool {
        filters.iter().any(|filter| {
            // Check graph filter
            if let Some(graph_filter) = &filter.graph_filter {
                if change.graph.as_ref() != Some(graph_filter) {
                    return false;
                }
            }
            
            // Check subject pattern
            if let Some(pattern) = &filter.subject_pattern {
                if let Some(subject) = &change.subject {
                    if !pattern.is_match(subject) {
                        return false;
                    }
                }
            }
            
            // Check predicate filter
            if let Some(pred_filter) = &filter.predicate_filter {
                if change.predicate.as_ref() != Some(pred_filter) {
                    return false;
                }
            }
            
            true
        })
    }
    
    /// Send notification to subscriber
    async fn send_to_subscriber(
        subscriber: &UpdateSubscriber,
        notification: UpdateNotification,
        config: &UpdateManagerConfig,
        stats: &Arc<RwLock<UpdateManagerStats>>,
    ) {
        match &subscriber.channel {
            UpdateChannel::WebSocket(sender) => {
                if let Err(e) = sender.send(notification).await {
                    warn!("Failed to send to WebSocket subscriber {}: {}", subscriber.id, e);
                } else {
                    stats.write().await.updates_sent += 1;
                }
            }
            UpdateChannel::Webhook { url, headers } => {
                // This would implement webhook delivery
                warn!("Webhook delivery not implemented yet");
            }
            _ => {
                warn!("Update channel not implemented yet");
            }
        }
    }
    
    /// Subscribe for updates
    pub async fn subscribe(
        &self,
        channel: UpdateChannel,
        filters: Vec<UpdateFilter>,
    ) -> Result<String> {
        let mut subscribers = self.subscribers.write().await;
        
        if subscribers.len() >= self.config.max_subscribers {
            return Err(anyhow!("Maximum subscriber limit reached"));
        }
        
        let id = uuid::Uuid::new_v4().to_string();
        let subscriber = UpdateSubscriber {
            id: id.clone(),
            channel,
            filters,
            created_at: Instant::now(),
            last_update: None,
            update_count: 0,
        };
        
        subscribers.insert(id.clone(), subscriber);
        
        self.stats.write().await.total_subscribers = subscribers.len();
        self.stats.write().await.active_subscribers = subscribers.len();
        
        Ok(id)
    }
    
    /// Unsubscribe from updates
    pub async fn unsubscribe(&self, id: &str) -> Result<()> {
        let mut subscribers = self.subscribers.write().await;
        subscribers.remove(id).ok_or_else(|| anyhow!("Subscriber not found"))?;
        
        self.stats.write().await.total_subscribers = subscribers.len();
        self.stats.write().await.active_subscribers = subscribers.len();
        
        Ok(())
    }
    
    /// Get statistics
    pub async fn get_stats(&self) -> UpdateManagerStats {
        self.stats.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Mock RDF store for testing
    struct MockRdfStore {
        log_position: Arc<RwLock<u64>>,
        changes: Arc<RwLock<Vec<TransactionLogEntry>>>,
    }
    
    #[async_trait::async_trait]
    impl RdfStore for MockRdfStore {
        async fn get_transaction_log_position(&self) -> Result<u64> {
            Ok(*self.log_position.read().await)
        }
        
        async fn read_transaction_log(&self, from: u64, limit: usize) -> Result<Vec<TransactionLogEntry>> {
            let changes = self.changes.read().await;
            Ok(changes.iter()
                .filter(|e| e.position >= from)
                .take(limit)
                .cloned()
                .collect())
        }
        
        async fn subscribe_changes(&self) -> Result<mpsc::Receiver<StoreChange>> {
            let (tx, rx) = mpsc::channel(100);
            Ok(rx)
        }
        
        async fn get_statistics(&self) -> Result<StoreStatistics> {
            Ok(StoreStatistics {
                total_triples: 1000,
                total_graphs: 10,
                transaction_count: 100,
                last_modified: chrono::Utc::now(),
            })
        }
        
        async fn apply_patch(&self, _patch: &RdfPatch) -> Result<()> {
            Ok(())
        }
        
        async fn execute_update(&self, _update: &str) -> Result<()> {
            Ok(())
        }
        
        async fn query(&self, _query: &str) -> Result<QueryResult> {
            Ok(QueryResult {
                bindings: vec![],
            })
        }
    }
    
    #[tokio::test]
    async fn test_change_detection() {
        let store = Arc::new(MockRdfStore {
            log_position: Arc::new(RwLock::new(0)),
            changes: Arc::new(RwLock::new(vec![])),
        });
        
        let stream_config = StreamConfig::memory();
        let producer = Arc::new(RwLock::new(
            StreamProducer::new(stream_config).await.unwrap()
        ));
        
        let strategy = ChangeDetectionStrategy::TransactionLog {
            poll_interval: Duration::from_millis(100),
            batch_size: 100,
        };
        
        let detector = StoreChangeDetector::new(
            store,
            strategy,
            producer,
            ChangeDetectorConfig::default(),
        ).await.unwrap();
        
        // Start detection
        detector.start().await.unwrap();
        
        // Verify statistics
        let stats = detector.get_stats().await;
        assert_eq!(stats.changes_detected, 0);
    }
}