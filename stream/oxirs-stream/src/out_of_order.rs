//! # Out-of-Order Event Handling Optimization
//!
//! This module provides advanced out-of-order event handling capabilities
//! for stream processing, ensuring correct event ordering even when events
//! arrive with varying delays.
//!
//! ## Features
//! - Event reordering buffers with configurable capacity
//! - Watermark-based late data handling
//! - Multiple strategies for late event processing
//! - Sequence number tracking and gap detection
//! - Configurable lateness tolerances
//! - Performance-optimized sorting algorithms
//!
//! ## Performance
//! - O(log n) insertion for sorted buffer
//! - Constant-time watermark updates
//! - Memory-efficient event storage

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::event::StreamEvent;

/// Configuration for out-of-order handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutOfOrderConfig {
    /// Maximum time to wait for late events
    pub max_lateness: Duration,
    /// Buffer capacity (number of events)
    pub buffer_capacity: usize,
    /// Strategy for handling late events
    pub late_event_strategy: LateEventStrategy,
    /// Enable watermark tracking
    pub enable_watermarks: bool,
    /// Watermark update interval
    pub watermark_interval: Duration,
    /// Allowed out-of-orderness
    pub allowed_out_of_orderness: Duration,
    /// Enable sequence number tracking
    pub enable_sequence_tracking: bool,
    /// Gap filling strategy
    pub gap_filling_strategy: GapFillingStrategy,
    /// Enable event deduplication
    pub enable_deduplication: bool,
    /// Deduplication window
    pub deduplication_window: Duration,
    /// Emit strategy
    pub emit_strategy: EmitStrategy,
}

impl Default for OutOfOrderConfig {
    fn default() -> Self {
        Self {
            max_lateness: Duration::from_secs(60),
            buffer_capacity: 10000,
            late_event_strategy: LateEventStrategy::SideOutput,
            enable_watermarks: true,
            watermark_interval: Duration::from_secs(1),
            allowed_out_of_orderness: Duration::from_secs(5),
            enable_sequence_tracking: true,
            gap_filling_strategy: GapFillingStrategy::Wait,
            enable_deduplication: true,
            deduplication_window: Duration::from_secs(60),
            emit_strategy: EmitStrategy::Watermark,
        }
    }
}

/// Strategy for handling late events
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LateEventStrategy {
    /// Drop late events silently
    Drop,
    /// Send to side output
    SideOutput,
    /// Reprocess with updated state
    Reprocess,
    /// Update aggregates without full reprocess
    UpdateOnly,
    /// Queue for manual review
    Queue,
}

/// Strategy for filling gaps in sequences
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GapFillingStrategy {
    /// Wait for missing events
    Wait,
    /// Skip gaps after timeout
    SkipAfterTimeout(Duration),
    /// Interpolate missing events
    Interpolate,
    /// Emit placeholder events
    Placeholder,
    /// Ignore gaps
    Ignore,
}

/// Strategy for emitting events
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EmitStrategy {
    /// Emit when watermark advances
    Watermark,
    /// Emit after fixed delay
    Delay(Duration),
    /// Emit when buffer is full
    BufferFull,
    /// Emit immediately with possible reordering
    Immediate,
    /// Emit after processing-time timeout
    ProcessingTimeTimeout(Duration),
}

/// Event with ordering information
#[derive(Debug, Clone)]
pub struct OrderedEvent {
    /// Original event
    pub event: StreamEvent,
    /// Event timestamp
    pub event_time: DateTime<Utc>,
    /// Sequence number
    pub sequence: Option<u64>,
    /// Ingestion time
    pub ingestion_time: DateTime<Utc>,
    /// Is this a late event
    pub is_late: bool,
    /// Gap before this event
    pub gap_before: Option<u64>,
}

/// Watermark for tracking event time progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Watermark {
    /// Current watermark timestamp
    pub timestamp: DateTime<Utc>,
    /// Last update time
    pub last_update: DateTime<Utc>,
    /// Number of events processed
    pub events_processed: u64,
}

impl Watermark {
    /// Create a new watermark
    pub fn new() -> Self {
        Self {
            timestamp: DateTime::from_timestamp(0, 0).unwrap(),
            last_update: Utc::now(),
            events_processed: 0,
        }
    }

    /// Update watermark with new event time
    pub fn update(&mut self, event_time: DateTime<Utc>, allowed_lateness: Duration) {
        // Watermark = event_time - allowed_lateness
        let lateness = chrono::Duration::from_std(allowed_lateness).unwrap_or_default();
        let new_watermark = event_time - lateness;

        if new_watermark > self.timestamp {
            self.timestamp = new_watermark;
            self.last_update = Utc::now();
        }
        self.events_processed += 1;
    }

    /// Check if event is late
    pub fn is_late(&self, event_time: DateTime<Utc>) -> bool {
        event_time < self.timestamp
    }
}

impl Default for Watermark {
    fn default() -> Self {
        Self::new()
    }
}

/// Out-of-order event handler
pub struct OutOfOrderHandler {
    /// Configuration
    config: OutOfOrderConfig,
    /// Event buffer (ordered by event time)
    buffer: Arc<RwLock<BTreeMap<i64, VecDeque<OrderedEvent>>>>,
    /// Current watermark
    watermark: Arc<RwLock<Watermark>>,
    /// Late events buffer
    late_events: Arc<RwLock<VecDeque<OrderedEvent>>>,
    /// Deduplication set
    seen_events: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
    /// Sequence tracker
    sequence_tracker: Arc<RwLock<SequenceTracker>>,
    /// Statistics
    stats: Arc<RwLock<OutOfOrderStats>>,
    /// Next emit time
    next_emit_time: Arc<RwLock<DateTime<Utc>>>,
}

/// Sequence tracker for gap detection
#[derive(Debug, Default)]
pub struct SequenceTracker {
    /// Expected next sequence
    expected_sequence: u64,
    /// Highest seen sequence
    highest_seen: u64,
    /// Missing sequences
    missing: Vec<u64>,
    /// Gaps detected
    gaps_detected: u64,
    /// Gaps filled
    gaps_filled: u64,
}

impl SequenceTracker {
    /// Track a sequence number
    pub fn track(&mut self, sequence: u64) -> Option<u64> {
        let gap = if sequence > self.expected_sequence {
            let gap_size = sequence - self.expected_sequence;
            for seq in self.expected_sequence..sequence {
                self.missing.push(seq);
            }
            self.gaps_detected += 1;
            Some(gap_size)
        } else {
            // Check if this fills a gap
            if let Some(pos) = self.missing.iter().position(|&s| s == sequence) {
                self.missing.remove(pos);
                self.gaps_filled += 1;
            }
            None
        };

        if sequence >= self.expected_sequence {
            self.expected_sequence = sequence + 1;
        }
        if sequence > self.highest_seen {
            self.highest_seen = sequence;
        }

        gap
    }

    /// Get missing sequences
    pub fn get_missing(&self) -> &[u64] {
        &self.missing
    }
}

/// Statistics for out-of-order handling
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutOfOrderStats {
    /// Total events processed
    pub total_events: u64,
    /// Events emitted in order
    pub ordered_events: u64,
    /// Late events detected
    pub late_events: u64,
    /// Late events dropped
    pub late_events_dropped: u64,
    /// Late events reprocessed
    pub late_events_reprocessed: u64,
    /// Duplicates detected
    pub duplicates_detected: u64,
    /// Gaps detected
    pub gaps_detected: u64,
    /// Gaps filled
    pub gaps_filled: u64,
    /// Current buffer size
    pub buffer_size: usize,
    /// Max buffer size reached
    pub max_buffer_size: usize,
    /// Average lateness
    pub avg_lateness_ms: f64,
    /// Max lateness
    pub max_lateness_ms: i64,
    /// Current watermark
    pub current_watermark: DateTime<Utc>,
    /// Events per second
    pub events_per_second: f64,
}

impl OutOfOrderHandler {
    /// Create a new out-of-order handler
    pub fn new(config: OutOfOrderConfig) -> Self {
        Self {
            config,
            buffer: Arc::new(RwLock::new(BTreeMap::new())),
            watermark: Arc::new(RwLock::new(Watermark::new())),
            late_events: Arc::new(RwLock::new(VecDeque::new())),
            seen_events: Arc::new(RwLock::new(HashMap::new())),
            sequence_tracker: Arc::new(RwLock::new(SequenceTracker::default())),
            stats: Arc::new(RwLock::new(OutOfOrderStats::default())),
            next_emit_time: Arc::new(RwLock::new(Utc::now())),
        }
    }

    /// Add an event to the handler
    pub async fn add_event(&self, event: StreamEvent) -> Result<Vec<OrderedEvent>> {
        let event_time = self.get_event_time(&event);
        let event_id = self.get_event_id(&event);
        let ingestion_time = Utc::now();

        // Check for duplicates
        if self.config.enable_deduplication {
            let mut seen = self.seen_events.write().await;
            if let Some(_first_seen) = seen.get(&event_id) {
                let mut stats = self.stats.write().await;
                stats.duplicates_detected += 1;
                debug!("Duplicate event detected: {}", event_id);
                return Ok(Vec::new());
            }
            seen.insert(event_id.clone(), ingestion_time);

            // Clean old entries
            let cutoff = ingestion_time
                - chrono::Duration::from_std(self.config.deduplication_window).unwrap_or_default();
            seen.retain(|_, time| *time > cutoff);
        }

        // Update watermark
        let is_late = if self.config.enable_watermarks {
            let mut watermark = self.watermark.write().await;
            let late = watermark.is_late(event_time);
            watermark.update(event_time, self.config.allowed_out_of_orderness);
            late
        } else {
            false
        };

        // Track sequence
        let (sequence, gap_before) = if self.config.enable_sequence_tracking {
            if let Some(seq) = self.get_sequence(&event) {
                let mut tracker = self.sequence_tracker.write().await;
                let gap = tracker.track(seq);
                (Some(seq), gap)
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        // Create ordered event
        let ordered_event = OrderedEvent {
            event,
            event_time,
            sequence,
            ingestion_time,
            is_late,
            gap_before,
        };

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_events += 1;

            let lateness_ms = (ingestion_time - event_time).num_milliseconds();
            stats.avg_lateness_ms = (stats.avg_lateness_ms * (stats.total_events - 1) as f64
                + lateness_ms as f64)
                / stats.total_events as f64;
            stats.max_lateness_ms = stats.max_lateness_ms.max(lateness_ms);

            if is_late {
                stats.late_events += 1;
            }
            if gap_before.is_some() {
                stats.gaps_detected += 1;
            }
        }

        // Handle late event
        if is_late {
            return self.handle_late_event(ordered_event).await;
        }

        // Add to buffer
        {
            let mut buffer = self.buffer.write().await;
            let timestamp_key = event_time.timestamp_millis();
            buffer
                .entry(timestamp_key)
                .or_insert_with(VecDeque::new)
                .push_back(ordered_event);

            let mut stats = self.stats.write().await;
            let total_size: usize = buffer.values().map(|v| v.len()).sum();
            stats.buffer_size = total_size;
            stats.max_buffer_size = stats.max_buffer_size.max(total_size);
        }

        // Emit events if ready
        self.emit_ready_events().await
    }

    /// Handle a late event based on configured strategy
    async fn handle_late_event(&self, event: OrderedEvent) -> Result<Vec<OrderedEvent>> {
        match &self.config.late_event_strategy {
            LateEventStrategy::Drop => {
                let mut stats = self.stats.write().await;
                stats.late_events_dropped += 1;
                debug!("Dropped late event: {:?}", event.event_time);
                Ok(Vec::new())
            }
            LateEventStrategy::SideOutput => {
                let mut late_events = self.late_events.write().await;
                late_events.push_back(event);

                // Trim if too many
                while late_events.len() > self.config.buffer_capacity / 10 {
                    late_events.pop_front();
                }

                Ok(Vec::new())
            }
            LateEventStrategy::Reprocess => {
                let mut stats = self.stats.write().await;
                stats.late_events_reprocessed += 1;
                Ok(vec![event])
            }
            LateEventStrategy::UpdateOnly => {
                let mut stats = self.stats.write().await;
                stats.late_events_reprocessed += 1;
                Ok(vec![event])
            }
            LateEventStrategy::Queue => {
                let mut late_events = self.late_events.write().await;
                late_events.push_back(event);
                Ok(Vec::new())
            }
        }
    }

    /// Emit events that are ready based on emit strategy
    async fn emit_ready_events(&self) -> Result<Vec<OrderedEvent>> {
        match &self.config.emit_strategy {
            EmitStrategy::Watermark => self.emit_before_watermark().await,
            EmitStrategy::Delay(delay) => self.emit_after_delay(*delay).await,
            EmitStrategy::BufferFull => self.emit_if_buffer_full().await,
            EmitStrategy::Immediate => self.emit_oldest().await,
            EmitStrategy::ProcessingTimeTimeout(timeout) => self.emit_after_timeout(*timeout).await,
        }
    }

    /// Emit events before current watermark
    async fn emit_before_watermark(&self) -> Result<Vec<OrderedEvent>> {
        let watermark = self.watermark.read().await.timestamp;
        let watermark_key = watermark.timestamp_millis();

        let mut buffer = self.buffer.write().await;
        let mut to_emit = Vec::new();

        // Collect all events before watermark
        let keys_to_remove: Vec<i64> = buffer.range(..watermark_key).map(|(k, _)| *k).collect();

        for key in keys_to_remove {
            if let Some(events) = buffer.remove(&key) {
                to_emit.extend(events);
            }
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.ordered_events += to_emit.len() as u64;
        stats.buffer_size = buffer.values().map(|v| v.len()).sum();
        stats.current_watermark = watermark;

        Ok(to_emit)
    }

    /// Emit events after a fixed delay
    async fn emit_after_delay(&self, delay: Duration) -> Result<Vec<OrderedEvent>> {
        let cutoff = Utc::now() - chrono::Duration::from_std(delay).unwrap_or_default();
        let cutoff_key = cutoff.timestamp_millis();

        let mut buffer = self.buffer.write().await;
        let mut to_emit = Vec::new();

        let keys_to_remove: Vec<i64> = buffer.range(..cutoff_key).map(|(k, _)| *k).collect();

        for key in keys_to_remove {
            if let Some(events) = buffer.remove(&key) {
                to_emit.extend(events);
            }
        }

        let mut stats = self.stats.write().await;
        stats.ordered_events += to_emit.len() as u64;
        stats.buffer_size = buffer.values().map(|v| v.len()).sum();

        Ok(to_emit)
    }

    /// Emit if buffer is full
    async fn emit_if_buffer_full(&self) -> Result<Vec<OrderedEvent>> {
        let buffer = self.buffer.read().await;
        let size: usize = buffer.values().map(|v| v.len()).sum();

        if size >= self.config.buffer_capacity {
            drop(buffer);
            // Emit oldest 10%
            let to_emit_count = self.config.buffer_capacity / 10;
            self.emit_n_oldest(to_emit_count).await
        } else {
            Ok(Vec::new())
        }
    }

    /// Emit oldest event
    async fn emit_oldest(&self) -> Result<Vec<OrderedEvent>> {
        self.emit_n_oldest(1).await
    }

    /// Emit N oldest events
    async fn emit_n_oldest(&self, n: usize) -> Result<Vec<OrderedEvent>> {
        let mut buffer = self.buffer.write().await;
        let mut to_emit = Vec::new();
        let mut remaining = n;

        while remaining > 0 {
            if let Some(first_key) = buffer.keys().next().copied() {
                if let Some(events) = buffer.get_mut(&first_key) {
                    while remaining > 0 && !events.is_empty() {
                        if let Some(event) = events.pop_front() {
                            to_emit.push(event);
                            remaining -= 1;
                        }
                    }
                    if events.is_empty() {
                        buffer.remove(&first_key);
                    }
                }
            } else {
                break;
            }
        }

        let mut stats = self.stats.write().await;
        stats.ordered_events += to_emit.len() as u64;
        stats.buffer_size = buffer.values().map(|v| v.len()).sum();

        Ok(to_emit)
    }

    /// Emit events after processing time timeout
    async fn emit_after_timeout(&self, timeout: Duration) -> Result<Vec<OrderedEvent>> {
        let now = Utc::now();
        let mut next_emit = self.next_emit_time.write().await;

        if now >= *next_emit {
            *next_emit = now + chrono::Duration::from_std(timeout).unwrap_or_default();
            drop(next_emit);

            // Emit all buffered events
            let mut buffer = self.buffer.write().await;
            let mut to_emit = Vec::new();

            for (_, events) in buffer.iter_mut() {
                to_emit.extend(events.drain(..));
            }
            buffer.clear();

            let mut stats = self.stats.write().await;
            stats.ordered_events += to_emit.len() as u64;
            stats.buffer_size = 0;

            Ok(to_emit)
        } else {
            Ok(Vec::new())
        }
    }

    /// Get event time from StreamEvent
    fn get_event_time(&self, event: &StreamEvent) -> DateTime<Utc> {
        match event {
            StreamEvent::TripleAdded { metadata, .. }
            | StreamEvent::TripleRemoved { metadata, .. }
            | StreamEvent::GraphCreated { metadata, .. }
            | StreamEvent::GraphDeleted { metadata, .. }
            | StreamEvent::TransactionBegin { metadata, .. }
            | StreamEvent::TransactionCommit { metadata, .. }
            | StreamEvent::TransactionAbort { metadata, .. }
            | StreamEvent::Heartbeat { metadata, .. }
            | StreamEvent::SparqlUpdate { metadata, .. }
            | StreamEvent::SchemaChanged { metadata, .. } => metadata.timestamp,
            _ => Utc::now(),
        }
    }

    /// Get event ID from StreamEvent
    fn get_event_id(&self, event: &StreamEvent) -> String {
        match event {
            StreamEvent::TripleAdded { metadata, .. }
            | StreamEvent::TripleRemoved { metadata, .. }
            | StreamEvent::GraphCreated { metadata, .. }
            | StreamEvent::GraphDeleted { metadata, .. }
            | StreamEvent::TransactionBegin { metadata, .. }
            | StreamEvent::TransactionCommit { metadata, .. }
            | StreamEvent::TransactionAbort { metadata, .. }
            | StreamEvent::Heartbeat { metadata, .. }
            | StreamEvent::SparqlUpdate { metadata, .. }
            | StreamEvent::SchemaChanged { metadata, .. } => metadata.event_id.clone(),
            _ => uuid::Uuid::new_v4().to_string(),
        }
    }

    /// Get sequence number from StreamEvent
    fn get_sequence(&self, event: &StreamEvent) -> Option<u64> {
        match event {
            StreamEvent::TripleAdded { metadata, .. }
            | StreamEvent::TripleRemoved { metadata, .. }
            | StreamEvent::Heartbeat { metadata, .. } => metadata
                .properties
                .get("sequence")
                .and_then(|s| s.parse().ok()),
            _ => None,
        }
    }

    /// Flush all buffered events
    pub async fn flush(&self) -> Result<Vec<OrderedEvent>> {
        let mut buffer = self.buffer.write().await;
        let mut to_emit = Vec::new();

        for (_, events) in buffer.iter_mut() {
            to_emit.extend(events.drain(..));
        }
        buffer.clear();

        let mut stats = self.stats.write().await;
        stats.ordered_events += to_emit.len() as u64;
        stats.buffer_size = 0;

        info!("Flushed {} events from out-of-order buffer", to_emit.len());

        Ok(to_emit)
    }

    /// Get late events
    pub async fn get_late_events(&self) -> Vec<OrderedEvent> {
        let late_events = self.late_events.read().await;
        late_events.iter().cloned().collect()
    }

    /// Clear late events
    pub async fn clear_late_events(&self) {
        let mut late_events = self.late_events.write().await;
        late_events.clear();
    }

    /// Get current watermark
    pub async fn get_watermark(&self) -> Watermark {
        self.watermark.read().await.clone()
    }

    /// Get statistics
    pub async fn get_stats(&self) -> OutOfOrderStats {
        self.stats.read().await.clone()
    }

    /// Get missing sequences
    pub async fn get_missing_sequences(&self) -> Vec<u64> {
        let tracker = self.sequence_tracker.read().await;
        tracker.get_missing().to_vec()
    }

    /// Reset handler state
    pub async fn reset(&self) {
        self.buffer.write().await.clear();
        self.late_events.write().await.clear();
        self.seen_events.write().await.clear();
        *self.watermark.write().await = Watermark::new();
        *self.sequence_tracker.write().await = SequenceTracker::default();
        *self.stats.write().await = OutOfOrderStats::default();

        info!("Out-of-order handler reset");
    }
}

/// Builder for out-of-order handler
pub struct OutOfOrderHandlerBuilder {
    config: OutOfOrderConfig,
}

impl OutOfOrderHandlerBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: OutOfOrderConfig::default(),
        }
    }

    /// Set maximum lateness
    pub fn max_lateness(mut self, duration: Duration) -> Self {
        self.config.max_lateness = duration;
        self
    }

    /// Set buffer capacity
    pub fn buffer_capacity(mut self, capacity: usize) -> Self {
        self.config.buffer_capacity = capacity;
        self
    }

    /// Set late event strategy
    pub fn late_event_strategy(mut self, strategy: LateEventStrategy) -> Self {
        self.config.late_event_strategy = strategy;
        self
    }

    /// Set allowed out-of-orderness
    pub fn allowed_out_of_orderness(mut self, duration: Duration) -> Self {
        self.config.allowed_out_of_orderness = duration;
        self
    }

    /// Set emit strategy
    pub fn emit_strategy(mut self, strategy: EmitStrategy) -> Self {
        self.config.emit_strategy = strategy;
        self
    }

    /// Enable deduplication
    pub fn with_deduplication(mut self, window: Duration) -> Self {
        self.config.enable_deduplication = true;
        self.config.deduplication_window = window;
        self
    }

    /// Enable sequence tracking
    pub fn with_sequence_tracking(mut self) -> Self {
        self.config.enable_sequence_tracking = true;
        self
    }

    /// Build the handler
    pub fn build(self) -> OutOfOrderHandler {
        OutOfOrderHandler::new(self.config)
    }
}

impl Default for OutOfOrderHandlerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventMetadata;
    use std::collections::HashMap;
    use uuid::Uuid;

    fn create_test_event(timestamp: DateTime<Utc>, sequence: Option<u64>) -> StreamEvent {
        let mut properties = HashMap::new();
        if let Some(seq) = sequence {
            properties.insert("sequence".to_string(), seq.to_string());
        }

        StreamEvent::TripleAdded {
            subject: "test:subject".to_string(),
            predicate: "test:predicate".to_string(),
            object: "test:object".to_string(),
            graph: None,
            metadata: EventMetadata {
                event_id: Uuid::new_v4().to_string(),
                timestamp,
                source: "test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties,
                checksum: None,
            },
        }
    }

    #[tokio::test]
    async fn test_handler_creation() {
        let handler = OutOfOrderHandler::new(OutOfOrderConfig::default());
        let stats = handler.get_stats().await;
        assert_eq!(stats.total_events, 0);
    }

    #[tokio::test]
    async fn test_add_event() {
        let handler = OutOfOrderHandlerBuilder::new()
            .emit_strategy(EmitStrategy::Immediate)
            .build();

        let event = create_test_event(Utc::now(), Some(1));
        let emitted = handler.add_event(event).await.unwrap();

        assert_eq!(emitted.len(), 1);
        let stats = handler.get_stats().await;
        assert_eq!(stats.total_events, 1);
    }

    #[tokio::test]
    async fn test_watermark_update() {
        let handler = OutOfOrderHandler::new(OutOfOrderConfig::default());

        let now = Utc::now();
        let event = create_test_event(now, None);
        handler.add_event(event).await.unwrap();

        let watermark = handler.get_watermark().await;
        assert!(watermark.events_processed > 0);
    }

    #[tokio::test]
    async fn test_late_event_detection() {
        let config = OutOfOrderConfig {
            late_event_strategy: LateEventStrategy::SideOutput,
            allowed_out_of_orderness: Duration::from_secs(1),
            ..Default::default()
        };
        let handler = OutOfOrderHandler::new(config);

        // Add a current event
        let now = Utc::now();
        let event = create_test_event(now, None);
        handler.add_event(event).await.unwrap();

        // Add a late event (10 seconds old)
        let old_time = now - chrono::Duration::seconds(10);
        let late_event = create_test_event(old_time, None);
        handler.add_event(late_event).await.unwrap();

        let stats = handler.get_stats().await;
        assert!(stats.late_events > 0);
    }

    #[tokio::test]
    async fn test_deduplication() {
        let config = OutOfOrderConfig {
            enable_deduplication: true,
            deduplication_window: Duration::from_secs(60),
            ..Default::default()
        };
        let handler = OutOfOrderHandler::new(config);

        let event = create_test_event(Utc::now(), None);
        let event_clone = event.clone();

        handler.add_event(event).await.unwrap();
        handler.add_event(event_clone).await.unwrap();

        let stats = handler.get_stats().await;
        assert_eq!(stats.duplicates_detected, 1);
    }

    #[tokio::test]
    async fn test_sequence_tracking() {
        let handler = OutOfOrderHandlerBuilder::new()
            .with_sequence_tracking()
            .emit_strategy(EmitStrategy::Immediate)
            .build();

        // Add events with gaps
        let now = Utc::now();
        handler
            .add_event(create_test_event(now, Some(1)))
            .await
            .unwrap();
        handler
            .add_event(create_test_event(now, Some(5)))
            .await
            .unwrap(); // Gap

        let missing = handler.get_missing_sequences().await;
        assert!(!missing.is_empty());
        assert!(missing.contains(&2));
        assert!(missing.contains(&3));
        assert!(missing.contains(&4));
    }

    #[tokio::test]
    async fn test_flush() {
        let handler = OutOfOrderHandlerBuilder::new()
            .emit_strategy(EmitStrategy::Watermark)
            .build();

        let now = Utc::now();
        handler
            .add_event(create_test_event(now, Some(1)))
            .await
            .unwrap();
        handler
            .add_event(create_test_event(now, Some(2)))
            .await
            .unwrap();

        let flushed = handler.flush().await.unwrap();
        assert!(flushed.len() >= 2);

        let stats = handler.get_stats().await;
        assert_eq!(stats.buffer_size, 0);
    }

    #[tokio::test]
    async fn test_emit_strategy_delay() {
        let handler = OutOfOrderHandlerBuilder::new()
            .emit_strategy(EmitStrategy::Delay(Duration::from_millis(100)))
            .build();

        let now = Utc::now();
        handler
            .add_event(create_test_event(now, Some(1)))
            .await
            .unwrap();

        // Should emit after delay
        tokio::time::sleep(Duration::from_millis(150)).await;
        let emitted = handler.emit_ready_events().await.unwrap();
        assert!(!emitted.is_empty());
    }

    #[tokio::test]
    async fn test_buffer_full_emit() {
        let handler = OutOfOrderHandlerBuilder::new()
            .buffer_capacity(10)
            .emit_strategy(EmitStrategy::BufferFull)
            .build();

        let now = Utc::now();
        for i in 0..15 {
            let time = now + chrono::Duration::milliseconds(i);
            handler
                .add_event(create_test_event(time, Some(i as u64)))
                .await
                .unwrap();
        }

        let stats = handler.get_stats().await;
        assert!(stats.ordered_events > 0);
    }

    #[tokio::test]
    async fn test_late_event_strategies() {
        // Test Drop strategy
        let handler = OutOfOrderHandlerBuilder::new()
            .late_event_strategy(LateEventStrategy::Drop)
            .allowed_out_of_orderness(Duration::from_secs(1))
            .emit_strategy(EmitStrategy::Immediate)
            .build();

        let now = Utc::now();
        handler
            .add_event(create_test_event(now, None))
            .await
            .unwrap();

        let old = now - chrono::Duration::seconds(100);
        let result = handler
            .add_event(create_test_event(old, None))
            .await
            .unwrap();
        assert!(result.is_empty());

        let stats = handler.get_stats().await;
        assert_eq!(stats.late_events_dropped, 1);
    }

    #[tokio::test]
    async fn test_reset() {
        let handler = OutOfOrderHandler::new(OutOfOrderConfig::default());

        let event = create_test_event(Utc::now(), Some(1));
        handler.add_event(event).await.unwrap();

        handler.reset().await;

        let stats = handler.get_stats().await;
        assert_eq!(stats.total_events, 0);
        assert_eq!(stats.buffer_size, 0);
    }

    #[tokio::test]
    async fn test_ordered_emission() {
        let handler = OutOfOrderHandlerBuilder::new()
            .emit_strategy(EmitStrategy::Delay(Duration::from_millis(50)))
            .build();

        // Add events out of order
        let base = Utc::now();
        handler
            .add_event(create_test_event(
                base + chrono::Duration::milliseconds(30),
                Some(3),
            ))
            .await
            .unwrap();
        handler
            .add_event(create_test_event(
                base + chrono::Duration::milliseconds(10),
                Some(1),
            ))
            .await
            .unwrap();
        handler
            .add_event(create_test_event(
                base + chrono::Duration::milliseconds(20),
                Some(2),
            ))
            .await
            .unwrap();

        // Wait and emit
        tokio::time::sleep(Duration::from_millis(100)).await;
        let emitted = handler.emit_ready_events().await.unwrap();

        // Verify ordering
        assert_eq!(emitted.len(), 3);
        for i in 0..emitted.len() - 1 {
            assert!(emitted[i].event_time <= emitted[i + 1].event_time);
        }
    }
}
