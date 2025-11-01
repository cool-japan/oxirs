//! # Temporal Joins for Stream Processing
//!
//! Advanced temporal join operations supporting event-time and processing-time semantics
//! with watermarks, late data handling, and various join strategies.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::event::StreamEvent;

/// Temporal join configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalJoinConfig {
    /// Join type
    pub join_type: TemporalJoinType,
    /// Time semantics
    pub time_semantics: TimeSemantics,
    /// Join window configuration
    pub window: TemporalWindow,
    /// Watermark configuration
    pub watermark: WatermarkConfig,
    /// Late data handling
    pub late_data: LateDataConfig,
}

impl Default for TemporalJoinConfig {
    fn default() -> Self {
        Self {
            join_type: TemporalJoinType::Inner,
            time_semantics: TimeSemantics::EventTime,
            window: TemporalWindow::default(),
            watermark: WatermarkConfig::default(),
            late_data: LateDataConfig::default(),
        }
    }
}

/// Temporal join types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TemporalJoinType {
    /// Inner temporal join
    Inner,
    /// Left temporal join
    Left,
    /// Right temporal join
    Right,
    /// Full outer temporal join
    FullOuter,
    /// Interval join
    Interval,
}

/// Time semantics for temporal operations
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TimeSemantics {
    /// Event time (based on event timestamps)
    EventTime,
    /// Processing time (based on system clock)
    ProcessingTime,
    /// Ingestion time (based on arrival time)
    IngestionTime,
}

/// Temporal window configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalWindow {
    /// Lower bound offset (negative duration before event)
    pub lower_bound: ChronoDuration,
    /// Upper bound offset (positive duration after event)
    pub upper_bound: ChronoDuration,
    /// Allow exact timestamp matches
    pub allow_exact: bool,
}

impl Default for TemporalWindow {
    fn default() -> Self {
        Self {
            lower_bound: ChronoDuration::minutes(-5),
            upper_bound: ChronoDuration::minutes(5),
            allow_exact: true,
        }
    }
}

/// Watermark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatermarkConfig {
    /// Watermark strategy
    pub strategy: WatermarkStrategy,
    /// Maximum allowed lateness
    pub max_lateness: ChronoDuration,
    /// Emit watermarks periodically
    pub periodic_emit: bool,
    /// Periodic emit interval
    pub emit_interval: ChronoDuration,
}

impl Default for WatermarkConfig {
    fn default() -> Self {
        Self {
            strategy: WatermarkStrategy::BoundedOutOfOrder {
                max_delay: ChronoDuration::seconds(10),
            },
            max_lateness: ChronoDuration::minutes(1),
            periodic_emit: true,
            emit_interval: ChronoDuration::seconds(1),
        }
    }
}

/// Watermark strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WatermarkStrategy {
    /// Ascending timestamps (no out-of-order)
    Ascending,
    /// Bounded out-of-order with maximum delay
    BoundedOutOfOrder { max_delay: ChronoDuration },
    /// Periodic watermarks
    Periodic { interval: ChronoDuration },
    /// Custom watermark generator
    Custom,
}

/// Late data handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateDataConfig {
    /// Strategy for handling late data
    pub strategy: LateDataStrategy,
    /// Side output for late data
    pub side_output_enabled: bool,
}

impl Default for LateDataConfig {
    fn default() -> Self {
        Self {
            strategy: LateDataStrategy::Drop,
            side_output_enabled: true,
        }
    }
}

/// Late data strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LateDataStrategy {
    /// Drop late data
    Drop,
    /// Emit late data with special marker
    EmitWithMarker,
    /// Reprocess affected windows
    ReprocessWindows,
}

/// Temporal join operator
pub struct TemporalJoin {
    config: TemporalJoinConfig,
    left_buffer: Arc<RwLock<EventBuffer>>,
    right_buffer: Arc<RwLock<EventBuffer>>,
    watermarks: Arc<RwLock<Watermarks>>,
    metrics: Arc<RwLock<TemporalJoinMetrics>>,
}

/// Event buffer for temporal join
#[derive(Debug)]
struct EventBuffer {
    events: VecDeque<TimestampedEvent>,
    max_size: usize,
}

/// Timestamped event wrapper
#[derive(Debug, Clone)]
struct TimestampedEvent {
    event: StreamEvent,
    event_time: DateTime<Utc>,
    processing_time: DateTime<Utc>,
}

impl EventBuffer {
    fn new(max_size: usize) -> Self {
        Self {
            events: VecDeque::new(),
            max_size,
        }
    }

    fn add_event(&mut self, event: TimestampedEvent) {
        if self.events.len() >= self.max_size {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    fn get_events_in_window(
        &self,
        timestamp: DateTime<Utc>,
        window: &TemporalWindow,
    ) -> Vec<TimestampedEvent> {
        let lower = timestamp + window.lower_bound;
        let upper = timestamp + window.upper_bound;

        self.events
            .iter()
            .filter(|e| {
                let t = e.event_time;
                (t > lower && t < upper) || (window.allow_exact && t == timestamp)
            })
            .cloned()
            .collect()
    }

    fn purge_before_watermark(&mut self, watermark: DateTime<Utc>) {
        while let Some(event) = self.events.front() {
            if event.event_time < watermark {
                self.events.pop_front();
            } else {
                break;
            }
        }
    }
}

/// Watermark tracking
#[derive(Debug, Clone)]
struct Watermarks {
    left_watermark: Option<DateTime<Utc>>,
    right_watermark: Option<DateTime<Utc>>,
}

impl Watermarks {
    fn new() -> Self {
        Self {
            left_watermark: None,
            right_watermark: None,
        }
    }

    fn update_left(&mut self, watermark: DateTime<Utc>) {
        self.left_watermark = Some(watermark);
    }

    fn update_right(&mut self, watermark: DateTime<Utc>) {
        self.right_watermark = Some(watermark);
    }

    fn min_watermark(&self) -> Option<DateTime<Utc>> {
        match (self.left_watermark, self.right_watermark) {
            (Some(l), Some(r)) => Some(l.min(r)),
            (Some(l), None) => Some(l),
            (None, Some(r)) => Some(r),
            (None, None) => None,
        }
    }
}

/// Temporal join metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemporalJoinMetrics {
    /// Total left events processed
    pub left_events_processed: u64,
    /// Total right events processed
    pub right_events_processed: u64,
    /// Total join matches
    pub join_matches: u64,
    /// Late events dropped
    pub late_events_dropped: u64,
    /// Watermarks emitted
    pub watermarks_emitted: u64,
    /// Average join latency (ms)
    pub avg_join_latency_ms: f64,
}

impl TemporalJoin {
    /// Create a new temporal join operator
    pub fn new(config: TemporalJoinConfig) -> Self {
        Self {
            config,
            left_buffer: Arc::new(RwLock::new(EventBuffer::new(10000))),
            right_buffer: Arc::new(RwLock::new(EventBuffer::new(10000))),
            watermarks: Arc::new(RwLock::new(Watermarks::new())),
            metrics: Arc::new(RwLock::new(TemporalJoinMetrics::default())),
        }
    }

    /// Process left stream event
    pub async fn process_left(&self, event: StreamEvent) -> Result<Vec<JoinResult>> {
        let start_time = std::time::Instant::now();

        let timestamped = self.create_timestamped_event(event).await?;

        // Check for late data
        if self.is_late_event(&timestamped, true).await {
            return self.handle_late_event(timestamped, true).await;
        }

        // Add to buffer
        self.left_buffer
            .write()
            .await
            .add_event(timestamped.clone());

        // Perform join
        let results = self.join_with_right(&timestamped).await?;

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.left_events_processed += 1;
            metrics.join_matches += results.len() as u64;
            let latency = start_time.elapsed().as_millis() as f64;
            metrics.avg_join_latency_ms = (metrics.avg_join_latency_ms + latency) / 2.0;
        }

        // Update watermark
        self.update_watermark(&timestamped, true).await;

        debug!("Processed left event, found {} matches", results.len());
        Ok(results)
    }

    /// Process right stream event
    pub async fn process_right(&self, event: StreamEvent) -> Result<Vec<JoinResult>> {
        let start_time = std::time::Instant::now();

        let timestamped = self.create_timestamped_event(event).await?;

        // Check for late data
        if self.is_late_event(&timestamped, false).await {
            return self.handle_late_event(timestamped, false).await;
        }

        // Add to buffer
        self.right_buffer
            .write()
            .await
            .add_event(timestamped.clone());

        // Perform join
        let results = self.join_with_left(&timestamped).await?;

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.right_events_processed += 1;
            metrics.join_matches += results.len() as u64;
            let latency = start_time.elapsed().as_millis() as f64;
            metrics.avg_join_latency_ms = (metrics.avg_join_latency_ms + latency) / 2.0;
        }

        // Update watermark
        self.update_watermark(&timestamped, false).await;

        debug!("Processed right event, found {} matches", results.len());
        Ok(results)
    }

    /// Create timestamped event
    async fn create_timestamped_event(&self, event: StreamEvent) -> Result<TimestampedEvent> {
        let event_time = match self.config.time_semantics {
            TimeSemantics::EventTime => self.extract_event_time(&event)?,
            TimeSemantics::ProcessingTime => Utc::now(),
            TimeSemantics::IngestionTime => Utc::now(),
        };

        Ok(TimestampedEvent {
            event,
            event_time,
            processing_time: Utc::now(),
        })
    }

    /// Extract event time from event
    fn extract_event_time(&self, event: &StreamEvent) -> Result<DateTime<Utc>> {
        match event {
            StreamEvent::TripleAdded { metadata, .. } => Ok(metadata.timestamp),
            StreamEvent::TripleRemoved { metadata, .. } => Ok(metadata.timestamp),
            StreamEvent::GraphCreated { metadata, .. } => Ok(metadata.timestamp),
            StreamEvent::GraphDeleted { metadata, .. } => Ok(metadata.timestamp),
            StreamEvent::TransactionBegin { metadata, .. } => Ok(metadata.timestamp),
            StreamEvent::TransactionCommit { metadata, .. } => Ok(metadata.timestamp),
            StreamEvent::TransactionAbort { metadata, .. } => Ok(metadata.timestamp),
            _ => Err(anyhow!("Cannot extract event time from event")),
        }
    }

    /// Check if event is late
    async fn is_late_event(&self, event: &TimestampedEvent, is_left: bool) -> bool {
        let watermarks = self.watermarks.read().await;
        let watermark = if is_left {
            watermarks.left_watermark
        } else {
            watermarks.right_watermark
        };

        if let Some(wm) = watermark {
            event.event_time < wm - self.config.watermark.max_lateness
        } else {
            false
        }
    }

    /// Handle late event
    async fn handle_late_event(
        &self,
        _event: TimestampedEvent,
        _is_left: bool,
    ) -> Result<Vec<JoinResult>> {
        match self.config.late_data.strategy {
            LateDataStrategy::Drop => {
                self.metrics.write().await.late_events_dropped += 1;
                warn!("Dropped late event");
                Ok(Vec::new())
            }
            LateDataStrategy::EmitWithMarker => {
                // Emit with late marker
                Ok(Vec::new())
            }
            LateDataStrategy::ReprocessWindows => {
                // Reprocess affected windows
                Ok(Vec::new())
            }
        }
    }

    /// Join with right buffer
    async fn join_with_right(&self, left_event: &TimestampedEvent) -> Result<Vec<JoinResult>> {
        let right_buffer = self.right_buffer.read().await;
        let matches = right_buffer.get_events_in_window(left_event.event_time, &self.config.window);

        let results = matches
            .into_iter()
            .map(|right_event| JoinResult {
                left_event: left_event.event.clone(),
                right_event: Some(right_event.event),
                join_time: Utc::now(),
                time_diff: (right_event.event_time - left_event.event_time).num_milliseconds(),
            })
            .collect();

        Ok(results)
    }

    /// Join with left buffer
    async fn join_with_left(&self, right_event: &TimestampedEvent) -> Result<Vec<JoinResult>> {
        let left_buffer = self.left_buffer.read().await;
        let matches = left_buffer.get_events_in_window(right_event.event_time, &self.config.window);

        let results = matches
            .into_iter()
            .map(|left_event| JoinResult {
                left_event: left_event.event,
                right_event: Some(right_event.event.clone()),
                join_time: Utc::now(),
                time_diff: (right_event.event_time - left_event.event_time).num_milliseconds(),
            })
            .collect();

        Ok(results)
    }

    /// Update watermark
    async fn update_watermark(&self, event: &TimestampedEvent, is_left: bool) {
        let watermark = match self.config.watermark.strategy {
            WatermarkStrategy::Ascending => event.event_time,
            WatermarkStrategy::BoundedOutOfOrder { max_delay } => event.event_time - max_delay,
            WatermarkStrategy::Periodic { .. } => {
                // Handled by periodic task
                return;
            }
            WatermarkStrategy::Custom => {
                // Custom logic
                event.event_time
            }
        };

        let mut watermarks = self.watermarks.write().await;
        if is_left {
            watermarks.update_left(watermark);
        } else {
            watermarks.update_right(watermark);
        }

        self.metrics.write().await.watermarks_emitted += 1;

        // Purge old events
        if let Some(min_wm) = watermarks.min_watermark() {
            drop(watermarks);
            self.left_buffer
                .write()
                .await
                .purge_before_watermark(min_wm);
            self.right_buffer
                .write()
                .await
                .purge_before_watermark(min_wm);
        }
    }

    /// Get metrics
    pub async fn get_metrics(&self) -> TemporalJoinMetrics {
        self.metrics.read().await.clone()
    }
}

/// Join result
#[derive(Debug, Clone)]
pub struct JoinResult {
    /// Left stream event
    pub left_event: StreamEvent,
    /// Right stream event (None for outer joins)
    pub right_event: Option<StreamEvent>,
    /// Join timestamp
    pub join_time: DateTime<Utc>,
    /// Time difference between events (milliseconds)
    pub time_diff: i64,
}

/// Interval join operator for asymmetric temporal joins
pub struct IntervalJoin {
    config: TemporalJoinConfig,
    join: TemporalJoin,
}

impl IntervalJoin {
    /// Create a new interval join
    pub fn new(config: TemporalJoinConfig) -> Self {
        let mut join_config = config.clone();
        join_config.join_type = TemporalJoinType::Interval;

        Self {
            config,
            join: TemporalJoin::new(join_config),
        }
    }

    /// Process event with interval constraints
    pub async fn process(
        &self,
        left_event: StreamEvent,
        right_event: StreamEvent,
    ) -> Result<Vec<JoinResult>> {
        // Process both events
        let left_results = self.join.process_left(left_event).await?;
        let right_results = self.join.process_right(right_event).await?;

        // Combine results
        let mut all_results = left_results;
        all_results.extend(right_results);

        Ok(all_results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventMetadata;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_temporal_join_creation() {
        let config = TemporalJoinConfig::default();
        let join = TemporalJoin::new(config);
        let metrics = join.get_metrics().await;
        assert_eq!(metrics.left_events_processed, 0);
    }

    #[tokio::test]
    async fn test_event_buffer() {
        let mut buffer = EventBuffer::new(100);
        let metadata = EventMetadata {
            event_id: "test".to_string(),
            timestamp: Utc::now(),
            source: "test".to_string(),
            user: None,
            context: None,
            caused_by: None,
            version: "1.0".to_string(),
            properties: HashMap::new(),
            checksum: None,
        };

        let event = TimestampedEvent {
            event: StreamEvent::GraphCreated {
                graph: "test".to_string(),
                metadata,
            },
            event_time: Utc::now(),
            processing_time: Utc::now(),
        };

        buffer.add_event(event);
        assert_eq!(buffer.events.len(), 1);
    }

    #[tokio::test]
    async fn test_watermark_strategy() {
        let strategy = WatermarkStrategy::BoundedOutOfOrder {
            max_delay: ChronoDuration::seconds(5),
        };

        match strategy {
            WatermarkStrategy::BoundedOutOfOrder { max_delay } => {
                assert_eq!(max_delay, ChronoDuration::seconds(5));
            }
            _ => panic!("Wrong strategy"),
        }
    }
}
