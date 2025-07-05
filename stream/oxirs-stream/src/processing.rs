//! # Advanced Event Processing
//!
//! Stream processing with windowing, aggregations, and complex event patterns.
//!
//! This module provides sophisticated event processing capabilities including:
//! - Time-based and count-based windowing
//! - Streaming aggregations (count, sum, average, min, max)
//! - Complex event pattern detection
//! - Event correlation and causality tracking
//! - Real-time analytics and metrics computation

use crate::event::StreamEventType;
use crate::{EventMetadata, StreamEvent};
use crate::quantum_processing::ComparisonOperator;
use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc, Timelike, Datelike};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::{Duration, Instant};
use tokio::time;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Window types for event processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowType {
    /// Fixed time-based window
    Tumbling { duration: ChronoDuration },
    /// Overlapping time-based window
    Sliding {
        duration: ChronoDuration,
        slide: ChronoDuration,
    },
    /// Count-based window
    CountBased { size: usize },
    /// Session-based window (events grouped by activity)
    Session { timeout: ChronoDuration },
    /// Custom window with user-defined logic
    Custom { name: String },
}

/// Watermark for tracking event time progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Watermark {
    /// Current watermark timestamp
    pub timestamp: DateTime<Utc>,
    /// Allowed lateness after watermark
    pub allowed_lateness: ChronoDuration,
}

impl Watermark {
    /// Create a new watermark with default values
    pub fn new() -> Self {
        Self {
            timestamp: Utc::now(),
            allowed_lateness: ChronoDuration::seconds(60),
        }
    }

    /// Update the watermark timestamp
    pub fn update(&mut self, timestamp: DateTime<Utc>) {
        if timestamp > self.timestamp {
            self.timestamp = timestamp;
        }
    }

    /// Get the current watermark timestamp
    pub fn current(&self) -> DateTime<Utc> {
        self.timestamp
    }
}

impl Default for Watermark {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregation functions for window processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregateFunction {
    Count,
    Sum { field: String },
    Average { field: String },
    Min { field: String },
    Max { field: String },
    First,
    Last,
    Distinct { field: String },
    Custom { name: String, expression: String },
}

/// Window configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowConfig {
    pub window_type: WindowType,
    pub aggregates: Vec<AggregateFunction>,
    pub group_by: Vec<String>,
    pub filter: Option<String>,
    pub allow_lateness: Option<ChronoDuration>,
    pub trigger: WindowTrigger,
}

/// Window trigger conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowTrigger {
    /// Trigger when window ends
    OnTime,
    /// Trigger every N events
    OnCount(usize),
    /// Trigger on specific conditions
    OnCondition(String),
    /// Trigger both on time and count
    Hybrid { time: ChronoDuration, count: usize },
}

/// Event processing window
#[derive(Debug)]
pub struct EventWindow {
    id: String,
    config: WindowConfig,
    events: VecDeque<StreamEvent>,
    start_time: DateTime<Utc>,
    end_time: Option<DateTime<Utc>>,
    last_trigger: Option<DateTime<Utc>>,
    event_count: usize,
    aggregation_state: HashMap<String, AggregationState>,
}

/// Aggregation state for maintaining running calculations
#[derive(Debug, Clone)]
enum AggregationState {
    Count(u64),
    Sum(f64),
    Average { sum: f64, count: u64 },
    Min(f64),
    Max(f64),
    First(StreamEvent),
    Last(StreamEvent),
    Distinct(std::collections::HashSet<String>),
}

/// Result of window aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowResult {
    pub window_id: String,
    pub window_start: DateTime<Utc>,
    pub window_end: DateTime<Utc>,
    pub event_count: usize,
    pub aggregations: HashMap<String, serde_json::Value>,
    pub trigger_reason: String,
    pub processing_time: DateTime<Utc>,
}

/// Advanced event processor with windowing and aggregations
pub struct EventProcessor {
    windows: HashMap<String, EventWindow>,
    watermark: DateTime<Utc>,
    late_events: VecDeque<(StreamEvent, DateTime<Utc>)>,
    stats: ProcessorStats,
    config: ProcessorConfig,
}

/// Processor configuration
#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    pub max_windows: usize,
    pub max_late_events: usize,
    pub watermark_delay: ChronoDuration,
    pub checkpoint_interval: Duration,
    pub enable_metrics: bool,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            max_windows: 1000,
            max_late_events: 10000,
            watermark_delay: ChronoDuration::seconds(5),
            checkpoint_interval: Duration::from_secs(60),
            enable_metrics: true,
        }
    }
}

/// Processor statistics
#[derive(Debug, Default)]
pub struct ProcessorStats {
    pub events_processed: u64,
    pub windows_created: u64,
    pub windows_triggered: u64,
    pub late_events: u64,
    pub dropped_events: u64,
    pub last_watermark: Option<DateTime<Utc>>,
    pub processing_latency_ms: f64,
}

impl EventProcessor {
    pub fn new() -> Self {
        Self {
            windows: HashMap::new(),
            watermark: Utc::now(),
            late_events: VecDeque::new(),
            stats: ProcessorStats::default(),
            config: ProcessorConfig::default(),
        }
    }

    pub fn with_config(mut self, config: ProcessorConfig) -> Self {
        self.config = config;
        self
    }

    /// Process a single event through all configured windows
    pub async fn process_event(&mut self, event: StreamEvent) -> Result<Vec<WindowResult>> {
        let start_time = Instant::now();
        let mut results = Vec::new();

        // Update stats
        self.stats.events_processed += 1;

        // Extract event timestamp
        let event_time = self.extract_event_time(&event);

        // Check if event is late
        if event_time < self.watermark - self.config.watermark_delay {
            self.handle_late_event(event, event_time).await?;
            return Ok(results);
        }

        // Process event through all windows
        let window_ids: Vec<String> = self.windows.keys().cloned().collect();
        for window_id in window_ids {
            if let Some(window_results) = self.add_event_to_window(&window_id, &event).await? {
                results.extend(window_results);
            }
        }

        // Update watermark
        self.update_watermark(event_time).await?;

        // Update processing latency
        self.stats.processing_latency_ms =
            (self.stats.processing_latency_ms + start_time.elapsed().as_millis() as f64) / 2.0;

        Ok(results)
    }

    /// Create a new window with the given configuration
    pub fn create_window(&mut self, config: WindowConfig) -> String {
        let window_id = Uuid::new_v4().to_string();
        let window = EventWindow::new(window_id.clone(), config);

        self.windows.insert(window_id.clone(), window);
        self.stats.windows_created += 1;

        info!("Created window: {}", window_id);
        window_id
    }

    /// Remove a window
    pub fn remove_window(&mut self, window_id: &str) -> Result<()> {
        if self.windows.remove(window_id).is_some() {
            info!("Removed window: {}", window_id);
            Ok(())
        } else {
            Err(anyhow!("Window not found: {}", window_id))
        }
    }

    /// Get processor statistics
    pub fn get_stats(&self) -> &ProcessorStats {
        &self.stats
    }

    /// Force trigger all windows
    pub async fn trigger_all_windows(&mut self) -> Result<Vec<WindowResult>> {
        let mut results = Vec::new();

        for (window_id, window) in &mut self.windows {
            if !window.events.is_empty() {
                if let Some(result) = window.trigger("manual").await? {
                    results.push(result);
                    self.stats.windows_triggered += 1;
                }
            }
        }

        Ok(results)
    }

    fn extract_event_time(&self, event: &StreamEvent) -> DateTime<Utc> {
        match event {
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
            | StreamEvent::ShapeAdded { metadata, .. }
            | StreamEvent::ShapeRemoved { metadata, .. }
            | StreamEvent::ShapeModified { metadata, .. }
            | StreamEvent::ShapeValidationStarted { metadata, .. }
            | StreamEvent::ShapeValidationCompleted { metadata, .. }
            | StreamEvent::ShapeViolationDetected { metadata, .. }
            | StreamEvent::QueryResultAdded { metadata, .. }
            | StreamEvent::QueryResultRemoved { metadata, .. }
            | StreamEvent::QueryCompleted { metadata, .. }
            | StreamEvent::SchemaUpdated { metadata, .. }
            | StreamEvent::ShapeUpdated { metadata, .. }
            | StreamEvent::ErrorOccurred { metadata, .. } => metadata.timestamp,
            StreamEvent::Heartbeat { timestamp, .. } => *timestamp,
        }
    }

    async fn handle_late_event(
        &mut self,
        event: StreamEvent,
        event_time: DateTime<Utc>,
    ) -> Result<()> {
        self.stats.late_events += 1;

        // Try to add to existing windows if they allow lateness
        let mut handled = false;
        let window_ids: Vec<String> = self.windows.keys().cloned().collect();

        for window_id in window_ids {
            if let Some(window) = self.windows.get(&window_id) {
                if let Some(allow_lateness) = &window.config.allow_lateness {
                    if event_time > window.start_time - *allow_lateness {
                        if let Some(_) = self.add_event_to_window(&window_id, &event).await? {
                            handled = true;
                            break;
                        }
                    }
                }
            }
        }

        if !handled {
            // Store in late events buffer
            self.late_events.push_back((event, event_time));

            // Trim buffer if too large
            while self.late_events.len() > self.config.max_late_events {
                self.late_events.pop_front();
                self.stats.dropped_events += 1;
            }

            warn!("Handled late event with timestamp: {}", event_time);
        }

        Ok(())
    }

    async fn add_event_to_window(
        &mut self,
        window_id: &str,
        event: &StreamEvent,
    ) -> Result<Option<Vec<WindowResult>>> {
        if let Some(window) = self.windows.get_mut(window_id) {
            window.add_event(event.clone()).await
        } else {
            Ok(None)
        }
    }

    async fn update_watermark(&mut self, event_time: DateTime<Utc>) -> Result<()> {
        let new_watermark = event_time - self.config.watermark_delay;

        if new_watermark > self.watermark {
            self.watermark = new_watermark;
            self.stats.last_watermark = Some(new_watermark);

            // Trigger any windows that should fire based on watermark
            let window_ids: Vec<String> = self.windows.keys().cloned().collect();
            for window_id in window_ids {
                if let Some(window) = self.windows.get_mut(&window_id) {
                    if let Some(end_time) = window.end_time {
                        if end_time <= self.watermark {
                            if let Some(_) = window.trigger("watermark").await? {
                                self.stats.windows_triggered += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl Default for EventProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl EventWindow {
    fn new(id: String, config: WindowConfig) -> Self {
        let start_time = Utc::now();
        let end_time = match &config.window_type {
            WindowType::Tumbling { duration } => Some(start_time + *duration),
            WindowType::Sliding { duration, .. } => Some(start_time + *duration),
            WindowType::Session { .. } => None, // Determined dynamically
            WindowType::CountBased { .. } => None, // No time-based end
            WindowType::Custom { .. } => None,  // Custom logic
        };

        Self {
            id,
            config,
            events: VecDeque::new(),
            start_time,
            end_time,
            last_trigger: None,
            event_count: 0,
            aggregation_state: HashMap::new(),
        }
    }

    async fn add_event(&mut self, event: StreamEvent) -> Result<Option<Vec<WindowResult>>> {
        self.events.push_back(event.clone());
        self.event_count += 1;

        // Update aggregation state
        self.update_aggregations(&event);

        // Check trigger conditions
        self.check_triggers().await
    }

    fn update_aggregations(&mut self, event: &StreamEvent) {
        for aggregate in &self.config.aggregates {
            let key = self.get_aggregate_key(aggregate);

            match aggregate {
                AggregateFunction::Count => {
                    let state = self
                        .aggregation_state
                        .entry(key)
                        .or_insert(AggregationState::Count(0));

                    if let AggregationState::Count(ref mut count) = state {
                        *count += 1;
                    }
                }
                AggregateFunction::Sum { field } => {
                    if let Some(value) = self.extract_numeric_field(event, field) {
                        let state = self
                            .aggregation_state
                            .entry(key)
                            .or_insert(AggregationState::Sum(0.0));

                        if let AggregationState::Sum(ref mut sum) = state {
                            *sum += value;
                        }
                    }
                }
                AggregateFunction::Average { field } => {
                    if let Some(value) = self.extract_numeric_field(event, field) {
                        let state = self
                            .aggregation_state
                            .entry(key)
                            .or_insert(AggregationState::Average { sum: 0.0, count: 0 });

                        if let AggregationState::Average {
                            ref mut sum,
                            ref mut count,
                        } = state
                        {
                            *sum += value;
                            *count += 1;
                        }
                    }
                }
                AggregateFunction::Min { field } => {
                    if let Some(value) = self.extract_numeric_field(event, field) {
                        let state = self
                            .aggregation_state
                            .entry(key.clone())
                            .or_insert(AggregationState::Min(f64::INFINITY));

                        if let AggregationState::Min(ref mut min_val) = state {
                            if value < *min_val {
                                *min_val = value;
                            }
                        }
                    }
                }
                AggregateFunction::Max { field } => {
                    if let Some(value) = self.extract_numeric_field(event, field) {
                        let state = self
                            .aggregation_state
                            .entry(key.clone())
                            .or_insert(AggregationState::Max(f64::NEG_INFINITY));

                        if let AggregationState::Max(ref mut max_val) = state {
                            if value > *max_val {
                                *max_val = value;
                            }
                        }
                    }
                }
                AggregateFunction::First => {
                    if !self.aggregation_state.contains_key(&key) {
                        self.aggregation_state
                            .insert(key, AggregationState::First(event.clone()));
                    }
                }
                AggregateFunction::Last => {
                    self.aggregation_state
                        .insert(key, AggregationState::Last(event.clone()));
                }
                AggregateFunction::Distinct { field } => {
                    if let Some(value) = self.extract_string_field(event, field) {
                        let state = self.aggregation_state.entry(key).or_insert(
                            AggregationState::Distinct(std::collections::HashSet::new()),
                        );

                        if let AggregationState::Distinct(ref mut set) = state {
                            set.insert(value);
                        }
                    }
                }
                AggregateFunction::Custom { .. } => {
                    // Custom aggregation would be implemented here
                    warn!("Custom aggregation not yet implemented");
                }
            }
        }
    }

    /// Generate consistent aggregation keys
    fn get_aggregate_key(&self, aggregate: &AggregateFunction) -> String {
        match aggregate {
            AggregateFunction::Count => "Count".to_string(),
            AggregateFunction::Sum { field } => format!("Sum {{ field: \"{}\" }}", field),
            AggregateFunction::Average { field } => format!("Average {{ field: \"{}\" }}", field),
            AggregateFunction::Min { field } => format!("Min {{ field: \"{}\" }}", field),
            AggregateFunction::Max { field } => format!("Max {{ field: \"{}\" }}", field),
            AggregateFunction::First => "First".to_string(),
            AggregateFunction::Last => "Last".to_string(),
            AggregateFunction::Distinct { field } => format!("Distinct {{ field: \"{}\" }}", field),
            AggregateFunction::Custom { name, .. } => format!("Custom {{ name: \"{}\" }}", name),
        }
    }

    fn extract_numeric_field(&self, event: &StreamEvent, field: &str) -> Option<f64> {
        // Extract numeric values from event fields
        match field {
            "count" => Some(1.0),
            "event_count" => Some(self.event_count as f64),
            _ => {
                // Try to extract from metadata properties
                match event {
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
                    | StreamEvent::ShapeAdded { metadata, .. }
                    | StreamEvent::ShapeRemoved { metadata, .. }
                    | StreamEvent::ShapeModified { metadata, .. }
                    | StreamEvent::ShapeValidationStarted { metadata, .. }
                    | StreamEvent::ShapeValidationCompleted { metadata, .. }
                    | StreamEvent::ShapeViolationDetected { metadata, .. }
                    | StreamEvent::QueryResultAdded { metadata, .. }
                    | StreamEvent::QueryResultRemoved { metadata, .. }
                    | StreamEvent::QueryCompleted { metadata, .. }
                    | StreamEvent::SchemaUpdated { metadata, .. }
                    | StreamEvent::ShapeUpdated { metadata, .. } => {
                        metadata.properties.get(field)?.parse().ok()
                    }
                    _ => None,
                }
            }
        }
    }

    fn extract_string_field(&self, event: &StreamEvent, field: &str) -> Option<String> {
        match field {
            "event_type" => Some(self.get_event_type(event)),
            "source" => Some(self.get_event_source(event)),
            _ => {
                // Try to extract from metadata properties
                match event {
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
                    | StreamEvent::ShapeAdded { metadata, .. }
                    | StreamEvent::ShapeRemoved { metadata, .. }
                    | StreamEvent::ShapeModified { metadata, .. }
                    | StreamEvent::ShapeValidationStarted { metadata, .. }
                    | StreamEvent::ShapeValidationCompleted { metadata, .. }
                    | StreamEvent::ShapeViolationDetected { metadata, .. }
                    | StreamEvent::QueryResultAdded { metadata, .. }
                    | StreamEvent::QueryResultRemoved { metadata, .. }
                    | StreamEvent::QueryCompleted { metadata, .. }
                    | StreamEvent::SchemaUpdated { metadata, .. }
                    | StreamEvent::ShapeUpdated { metadata, .. } => {
                        metadata.properties.get(field).cloned()
                    }
                    _ => None,
                }
            }
        }
    }

    fn get_event_type(&self, event: &StreamEvent) -> String {
        match event {
            StreamEvent::TripleAdded { .. } => "triple_added".to_string(),
            StreamEvent::TripleRemoved { .. } => "triple_removed".to_string(),
            StreamEvent::QuadAdded { .. } => "quad_added".to_string(),
            StreamEvent::QuadRemoved { .. } => "quad_removed".to_string(),
            StreamEvent::GraphCreated { .. } => "graph_created".to_string(),
            StreamEvent::GraphCleared { .. } => "graph_cleared".to_string(),
            StreamEvent::GraphDeleted { .. } => "graph_deleted".to_string(),
            StreamEvent::GraphMetadataUpdated { .. } => "graph_metadata_updated".to_string(),
            StreamEvent::GraphPermissionsChanged { .. } => "graph_permissions_changed".to_string(),
            StreamEvent::GraphStatisticsUpdated { .. } => "graph_statistics_updated".to_string(),
            StreamEvent::GraphRenamed { .. } => "graph_renamed".to_string(),
            StreamEvent::GraphMerged { .. } => "graph_merged".to_string(),
            StreamEvent::GraphSplit { .. } => "graph_split".to_string(),
            StreamEvent::SparqlUpdate { .. } => "sparql_update".to_string(),
            StreamEvent::TransactionBegin { .. } => "transaction_begin".to_string(),
            StreamEvent::TransactionCommit { .. } => "transaction_commit".to_string(),
            StreamEvent::TransactionAbort { .. } => "transaction_abort".to_string(),
            StreamEvent::SchemaChanged { .. } => "schema_changed".to_string(),
            StreamEvent::SchemaDefinitionAdded { .. } => "schema_definition_added".to_string(),
            StreamEvent::SchemaDefinitionRemoved { .. } => "schema_definition_removed".to_string(),
            StreamEvent::SchemaDefinitionModified { .. } => {
                "schema_definition_modified".to_string()
            }
            StreamEvent::OntologyImported { .. } => "ontology_imported".to_string(),
            StreamEvent::OntologyRemoved { .. } => "ontology_removed".to_string(),
            StreamEvent::ConstraintAdded { .. } => "constraint_added".to_string(),
            StreamEvent::ConstraintRemoved { .. } => "constraint_removed".to_string(),
            StreamEvent::ConstraintViolated { .. } => "constraint_violated".to_string(),
            StreamEvent::IndexCreated { .. } => "index_created".to_string(),
            StreamEvent::IndexDropped { .. } => "index_dropped".to_string(),
            StreamEvent::IndexRebuilt { .. } => "index_rebuilt".to_string(),
            StreamEvent::ShapeAdded { .. } => "shape_added".to_string(),
            StreamEvent::ShapeRemoved { .. } => "shape_removed".to_string(),
            StreamEvent::ShapeModified { .. } => "shape_modified".to_string(),
            StreamEvent::ShapeValidationStarted { .. } => "shape_validation_started".to_string(),
            StreamEvent::ShapeValidationCompleted { .. } => {
                "shape_validation_completed".to_string()
            }
            StreamEvent::ShapeViolationDetected { .. } => "shape_violation_detected".to_string(),
            StreamEvent::QueryResultAdded { .. } => "query_result_added".to_string(),
            StreamEvent::QueryResultRemoved { .. } => "query_result_removed".to_string(),
            StreamEvent::QueryCompleted { .. } => "query_completed".to_string(),
            StreamEvent::SchemaUpdated { .. } => "schema_updated".to_string(),
            StreamEvent::ShapeUpdated { .. } => "shape_updated".to_string(),
            StreamEvent::Heartbeat { .. } => "heartbeat".to_string(),
            StreamEvent::ErrorOccurred { .. } => "error_occurred".to_string(),
        }
    }

    fn get_event_source(&self, event: &StreamEvent) -> String {
        match event {
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
            | StreamEvent::ShapeAdded { metadata, .. }
            | StreamEvent::ShapeRemoved { metadata, .. }
            | StreamEvent::ShapeModified { metadata, .. }
            | StreamEvent::ShapeValidationStarted { metadata, .. }
            | StreamEvent::ShapeValidationCompleted { metadata, .. }
            | StreamEvent::ShapeViolationDetected { metadata, .. }
            | StreamEvent::QueryResultAdded { metadata, .. }
            | StreamEvent::QueryResultRemoved { metadata, .. }
            | StreamEvent::QueryCompleted { metadata, .. }
            | StreamEvent::SchemaUpdated { metadata, .. }
            | StreamEvent::ShapeUpdated { metadata, .. }
            | StreamEvent::ErrorOccurred { metadata, .. } => metadata.source.clone(),
            StreamEvent::Heartbeat { source, .. } => source.clone(),
        }
    }

    async fn check_triggers(&mut self) -> Result<Option<Vec<WindowResult>>> {
        let should_trigger = match &self.config.trigger {
            WindowTrigger::OnTime => {
                if let Some(end_time) = self.end_time {
                    Utc::now() >= end_time
                } else {
                    false
                }
            }
            WindowTrigger::OnCount(count) => self.event_count >= *count,
            WindowTrigger::OnCondition(_condition) => {
                // Custom condition evaluation would go here
                false
            }
            WindowTrigger::Hybrid { time, count } => {
                let time_trigger = if let Some(end_time) = self.end_time {
                    Utc::now() >= end_time
                } else {
                    Utc::now() >= self.start_time + *time
                };
                let count_trigger = self.event_count >= *count;
                time_trigger || count_trigger
            }
        };

        if should_trigger {
            if let Some(result) = self.trigger("condition").await? {
                Ok(Some(vec![result]))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    async fn trigger(&mut self, reason: &str) -> Result<Option<WindowResult>> {
        if self.events.is_empty() {
            return Ok(None);
        }

        let mut aggregations = HashMap::new();

        for (key, state) in &self.aggregation_state {
            let value = match state {
                AggregationState::Count(count) => {
                    serde_json::Value::Number(serde_json::Number::from(*count))
                }
                AggregationState::Sum(sum) => serde_json::Value::Number(
                    serde_json::Number::from_f64(*sum)
                        .unwrap_or_else(|| serde_json::Number::from(0)),
                ),
                AggregationState::Average { sum, count } => {
                    let avg = if *count > 0 {
                        sum / (*count as f64)
                    } else {
                        0.0
                    };
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(avg)
                            .unwrap_or_else(|| serde_json::Number::from(0)),
                    )
                }
                AggregationState::Min(min) => {
                    if *min == f64::INFINITY {
                        serde_json::Value::Null // No values seen yet
                    } else {
                        serde_json::Value::Number(
                            serde_json::Number::from_f64(*min)
                                .unwrap_or_else(|| serde_json::Number::from(0)),
                        )
                    }
                }
                AggregationState::Max(max) => {
                    if *max == f64::NEG_INFINITY {
                        serde_json::Value::Null // No values seen yet
                    } else {
                        serde_json::Value::Number(
                            serde_json::Number::from_f64(*max)
                                .unwrap_or_else(|| serde_json::Number::from(0)),
                        )
                    }
                }
                AggregationState::First(_) => serde_json::Value::String("first_event".to_string()),
                AggregationState::Last(_) => serde_json::Value::String("last_event".to_string()),
                AggregationState::Distinct(set) => {
                    serde_json::Value::Number(serde_json::Number::from(set.len() as u64))
                }
            };

            aggregations.insert(key.clone(), value);
        }

        let result = WindowResult {
            window_id: self.id.clone(),
            window_start: self.start_time,
            window_end: self.end_time.unwrap_or_else(Utc::now),
            event_count: self.event_count,
            aggregations,
            trigger_reason: reason.to_string(),
            processing_time: Utc::now(),
        };

        // Reset window state for sliding windows
        match &self.config.window_type {
            WindowType::Sliding { slide, .. } => {
                self.start_time = self.start_time + *slide;
                if let Some(end_time) = self.end_time {
                    self.end_time = Some(end_time + *slide);
                }
                // Keep events that are still within the new window
                // For simplicity, we're clearing here - in a real implementation
                // we'd keep relevant events
                self.events.clear();
                self.event_count = 0;
                self.aggregation_state.clear();
            }
            _ => {
                // For other window types, typically we'd close this window
                // and create a new one, but here we'll just reset
                self.events.clear();
                self.event_count = 0;
                self.aggregation_state.clear();
            }
        }

        self.last_trigger = Some(Utc::now());

        debug!(
            "Triggered window {} with {} events",
            self.id, result.event_count
        );
        Ok(Some(result))
    }
}

/// Pattern detection for complex event processing
#[derive(Debug, Clone)]
pub struct EventPattern {
    pub name: String,
    pub conditions: Vec<PatternCondition>,
    pub within: Option<ChronoDuration>,
    pub action: PatternAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternCondition {
    EventType(String),
    FieldEquals { field: String, value: String },
    FieldGreater { field: String, value: f64 },
    FieldLess { field: String, value: f64 },
    Sequence(Vec<PatternCondition>),
    Any(Vec<PatternCondition>),
    All(Vec<PatternCondition>),
}

#[derive(Debug, Clone)]
pub enum PatternAction {
    Log(String),
    Alert { severity: String, message: String },
    Emit(StreamEvent),
    Custom(String),
}

/// Complex event processor for pattern detection
pub struct ComplexEventProcessor {
    patterns: Vec<EventPattern>,
    event_buffer: VecDeque<(StreamEvent, DateTime<Utc>)>,
    buffer_size: usize,
    matches: Vec<PatternMatch>,
}

#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern_name: String,
    pub matched_events: Vec<StreamEvent>,
    pub match_time: DateTime<Utc>,
    pub confidence: f64,
}

impl ComplexEventProcessor {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            event_buffer: VecDeque::new(),
            buffer_size: 1000,
            matches: Vec::new(),
        }
    }

    pub fn add_pattern(&mut self, pattern: EventPattern) {
        self.patterns.push(pattern);
        info!("Added pattern: {}", self.patterns.last().unwrap().name);
    }

    pub async fn process_event(&mut self, event: StreamEvent) -> Result<Vec<PatternMatch>> {
        let event_time = self.extract_event_time(&event);

        // Add to buffer
        self.event_buffer.push_back((event.clone(), event_time));

        // Trim buffer if too large
        if self.event_buffer.len() > self.buffer_size {
            self.event_buffer.pop_front();
        }

        // Check patterns
        let mut matches = Vec::new();
        for pattern in &self.patterns {
            if let Some(pattern_match) = self.check_pattern(pattern, &event, event_time).await? {
                matches.push(pattern_match);
            }
        }

        self.matches.extend(matches.clone());
        Ok(matches)
    }

    fn extract_event_time(&self, event: &StreamEvent) -> DateTime<Utc> {
        match event {
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
            | StreamEvent::QueryResultAdded { metadata, .. }
            | StreamEvent::QueryResultRemoved { metadata, .. }
            | StreamEvent::QueryCompleted { metadata, .. } => metadata.timestamp,
            StreamEvent::Heartbeat { timestamp, .. } => *timestamp,
            // For variants without metadata, use current time as fallback
            _ => Utc::now(),
        }
    }

    async fn check_pattern(
        &self,
        pattern: &EventPattern,
        current_event: &StreamEvent,
        event_time: DateTime<Utc>,
    ) -> Result<Option<PatternMatch>> {
        // Filter events within the time window if specified
        let relevant_events = if let Some(within) = &pattern.within {
            let cutoff_time = event_time - *within;
            self.event_buffer
                .iter()
                .filter(|(_, t)| *t >= cutoff_time && *t <= event_time)
                .map(|(e, _)| e.clone())
                .collect::<Vec<_>>()
        } else {
            self.event_buffer
                .iter()
                .map(|(e, _)| e.clone())
                .collect::<Vec<_>>()
        };

        // Check if the pattern conditions match
        let mut matched_events = Vec::new();
        let confidence = self.evaluate_conditions(
            &pattern.conditions,
            &relevant_events,
            current_event,
            &mut matched_events,
        )?;

        if confidence > 0.0 {
            // Execute the pattern action
            self.execute_action(&pattern.action, &matched_events)
                .await?;

            Ok(Some(PatternMatch {
                pattern_name: pattern.name.clone(),
                matched_events,
                match_time: event_time,
                confidence,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn get_matches(&self) -> &[PatternMatch] {
        &self.matches
    }

    /// Evaluate pattern conditions against events
    fn evaluate_conditions(
        &self,
        conditions: &[PatternCondition],
        relevant_events: &[StreamEvent],
        current_event: &StreamEvent,
        matched_events: &mut Vec<StreamEvent>,
    ) -> Result<f64> {
        if conditions.is_empty() {
            return Ok(0.0);
        }

        let mut total_confidence = 0.0;
        let mut matches = 0;

        for condition in conditions {
            match condition {
                PatternCondition::EventType(expected_type) => {
                    if self.match_event_type(current_event, expected_type) {
                        matched_events.push(current_event.clone());
                        matches += 1;
                        total_confidence += 1.0;
                    }
                }
                PatternCondition::FieldEquals { field, value } => {
                    if self.match_field_equals(current_event, field, value) {
                        matched_events.push(current_event.clone());
                        matches += 1;
                        total_confidence += 1.0;
                    }
                }
                PatternCondition::FieldGreater { field, value } => {
                    if self.match_field_greater(current_event, field, *value) {
                        matched_events.push(current_event.clone());
                        matches += 1;
                        total_confidence += 1.0;
                    }
                }
                PatternCondition::FieldLess { field, value } => {
                    if self.match_field_less(current_event, field, *value) {
                        matched_events.push(current_event.clone());
                        matches += 1;
                        total_confidence += 1.0;
                    }
                }
                PatternCondition::Sequence(seq_conditions) => {
                    let seq_confidence =
                        self.evaluate_sequence(seq_conditions, relevant_events, matched_events)?;
                    if seq_confidence > 0.0 {
                        matches += 1;
                        total_confidence += seq_confidence;
                    }
                }
                PatternCondition::Any(any_conditions) => {
                    let any_confidence = self.evaluate_any(
                        any_conditions,
                        relevant_events,
                        current_event,
                        matched_events,
                    )?;
                    if any_confidence > 0.0 {
                        matches += 1;
                        total_confidence += any_confidence;
                    }
                }
                PatternCondition::All(all_conditions) => {
                    let all_confidence = self.evaluate_all(
                        all_conditions,
                        relevant_events,
                        current_event,
                        matched_events,
                    )?;
                    if all_confidence > 0.0 {
                        matches += 1;
                        total_confidence += all_confidence;
                    }
                }
            }
        }

        if matches > 0 {
            Ok(total_confidence / conditions.len() as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Execute pattern action
    async fn execute_action(
        &self,
        action: &PatternAction,
        matched_events: &[StreamEvent],
    ) -> Result<()> {
        match action {
            PatternAction::Log(message) => {
                info!(
                    "Pattern matched: {} ({} events)",
                    message,
                    matched_events.len()
                );
            }
            PatternAction::Alert { severity, message } => {
                warn!(
                    "Pattern alert [{}]: {} ({} events)",
                    severity,
                    message,
                    matched_events.len()
                );
            }
            PatternAction::Emit(_event) => {
                // In a real implementation, this would emit a new event to a stream
                debug!("Emitting event for pattern match");
            }
            PatternAction::Custom(action_name) => {
                debug!("Executing custom action: {}", action_name);
            }
        }
        Ok(())
    }

    /// Match event type
    fn match_event_type(&self, event: &StreamEvent, expected_type: &str) -> bool {
        let event_type = match event {
            StreamEvent::TripleAdded { .. } => "triple_added",
            StreamEvent::TripleRemoved { .. } => "triple_removed",
            StreamEvent::QuadAdded { .. } => "quad_added",
            StreamEvent::QuadRemoved { .. } => "quad_removed",
            StreamEvent::GraphCreated { .. } => "graph_created",
            StreamEvent::GraphCleared { .. } => "graph_cleared",
            StreamEvent::GraphDeleted { .. } => "graph_deleted",
            StreamEvent::GraphMetadataUpdated { .. } => "graph_metadata_updated",
            StreamEvent::GraphPermissionsChanged { .. } => "graph_permissions_changed",
            StreamEvent::GraphStatisticsUpdated { .. } => "graph_statistics_updated",
            StreamEvent::GraphRenamed { .. } => "graph_renamed",
            StreamEvent::GraphMerged { .. } => "graph_merged",
            StreamEvent::GraphSplit { .. } => "graph_split",
            StreamEvent::SparqlUpdate { .. } => "sparql_update",
            StreamEvent::TransactionBegin { .. } => "transaction_begin",
            StreamEvent::TransactionCommit { .. } => "transaction_commit",
            StreamEvent::TransactionAbort { .. } => "transaction_abort",
            StreamEvent::SchemaChanged { .. } => "schema_changed",
            StreamEvent::SchemaDefinitionAdded { .. } => "schema_definition_added",
            StreamEvent::SchemaDefinitionRemoved { .. } => "schema_definition_removed",
            StreamEvent::SchemaDefinitionModified { .. } => "schema_definition_modified",
            StreamEvent::OntologyImported { .. } => "ontology_imported",
            StreamEvent::OntologyRemoved { .. } => "ontology_removed",
            StreamEvent::ConstraintAdded { .. } => "constraint_added",
            StreamEvent::ConstraintRemoved { .. } => "constraint_removed",
            StreamEvent::ConstraintViolated { .. } => "constraint_violated",
            StreamEvent::IndexCreated { .. } => "index_created",
            StreamEvent::IndexDropped { .. } => "index_dropped",
            StreamEvent::IndexRebuilt { .. } => "index_rebuilt",
            StreamEvent::ShapeAdded { .. } => "shape_added",
            StreamEvent::ShapeRemoved { .. } => "shape_removed",
            StreamEvent::ShapeModified { .. } => "shape_modified",
            StreamEvent::ShapeValidationStarted { .. } => "shape_validation_started",
            StreamEvent::ShapeValidationCompleted { .. } => "shape_validation_completed",
            StreamEvent::ShapeViolationDetected { .. } => "shape_violation_detected",
            StreamEvent::QueryResultAdded { .. } => "query_result_added",
            StreamEvent::QueryResultRemoved { .. } => "query_result_removed",
            StreamEvent::QueryCompleted { .. } => "query_completed",
            StreamEvent::SchemaUpdated { .. } => "schema_updated",
            StreamEvent::ShapeUpdated { .. } => "shape_updated",
            StreamEvent::Heartbeat { .. } => "heartbeat",
            StreamEvent::ErrorOccurred { .. } => "error_occurred",
        };
        event_type == expected_type
    }

    /// Match field equals condition
    fn match_field_equals(&self, event: &StreamEvent, field: &str, expected_value: &str) -> bool {
        match field {
            "subject" => match event {
                StreamEvent::TripleAdded { subject, .. }
                | StreamEvent::TripleRemoved { subject, .. }
                | StreamEvent::QuadAdded { subject, .. }
                | StreamEvent::QuadRemoved { subject, .. } => subject == expected_value,
                _ => false,
            },
            "predicate" => match event {
                StreamEvent::TripleAdded { predicate, .. }
                | StreamEvent::TripleRemoved { predicate, .. }
                | StreamEvent::QuadAdded { predicate, .. }
                | StreamEvent::QuadRemoved { predicate, .. } => predicate == expected_value,
                _ => false,
            },
            "object" => match event {
                StreamEvent::TripleAdded { object, .. }
                | StreamEvent::TripleRemoved { object, .. }
                | StreamEvent::QuadAdded { object, .. }
                | StreamEvent::QuadRemoved { object, .. } => object == expected_value,
                _ => false,
            },
            "graph" => match event {
                StreamEvent::TripleAdded { graph, .. }
                | StreamEvent::TripleRemoved { graph, .. } => {
                    graph.as_ref().map(|g| g == expected_value).unwrap_or(false)
                }
                StreamEvent::QuadAdded { graph, .. } | StreamEvent::QuadRemoved { graph, .. } => {
                    graph == expected_value
                }
                StreamEvent::GraphCreated { graph, .. }
                | StreamEvent::GraphDeleted { graph, .. } => graph == expected_value,
                StreamEvent::GraphCleared { graph, .. } => {
                    graph.as_ref().map(|g| g == expected_value).unwrap_or(false)
                }
                _ => false,
            },
            _ => {
                // Check metadata properties
                self.get_event_property(event, field)
                    .map(|v| v == expected_value)
                    .unwrap_or(false)
            }
        }
    }

    /// Match field greater than condition
    fn match_field_greater(&self, event: &StreamEvent, field: &str, threshold: f64) -> bool {
        self.get_event_numeric_property(event, field)
            .map(|v| v > threshold)
            .unwrap_or(false)
    }

    /// Match field less than condition
    fn match_field_less(&self, event: &StreamEvent, field: &str, threshold: f64) -> bool {
        self.get_event_numeric_property(event, field)
            .map(|v| v < threshold)
            .unwrap_or(false)
    }

    /// Evaluate sequence condition
    fn evaluate_sequence(
        &self,
        conditions: &[PatternCondition],
        events: &[StreamEvent],
        matched_events: &mut Vec<StreamEvent>,
    ) -> Result<f64> {
        if conditions.is_empty() || events.len() < conditions.len() {
            return Ok(0.0);
        }

        // Simple sequence matching - check if conditions match in order
        let mut condition_idx = 0;
        let mut sequence_matches = Vec::new();

        for event in events {
            if condition_idx >= conditions.len() {
                break;
            }

            let mut temp_matches = Vec::new();
            let confidence = self.evaluate_conditions(
                &[conditions[condition_idx].clone()],
                &[event.clone()],
                event,
                &mut temp_matches,
            )?;

            if confidence > 0.0 {
                sequence_matches.push(event.clone());
                condition_idx += 1;
            }
        }

        if condition_idx == conditions.len() {
            matched_events.extend(sequence_matches);
            Ok(1.0)
        } else {
            Ok(0.0)
        }
    }

    /// Evaluate any condition (OR)
    fn evaluate_any(
        &self,
        conditions: &[PatternCondition],
        events: &[StreamEvent],
        current_event: &StreamEvent,
        matched_events: &mut Vec<StreamEvent>,
    ) -> Result<f64> {
        let mut max_confidence = 0.0;

        for condition in conditions {
            let mut temp_matches = Vec::new();
            let confidence = self.evaluate_conditions(
                &[condition.clone()],
                events,
                current_event,
                &mut temp_matches,
            )?;

            if confidence > max_confidence {
                max_confidence = confidence;
                matched_events.clear();
                matched_events.extend(temp_matches);
            }
        }

        Ok(max_confidence)
    }

    /// Evaluate all conditions (AND)
    fn evaluate_all(
        &self,
        conditions: &[PatternCondition],
        events: &[StreamEvent],
        current_event: &StreamEvent,
        matched_events: &mut Vec<StreamEvent>,
    ) -> Result<f64> {
        let mut total_confidence = 0.0;
        let mut all_matches = Vec::new();

        for condition in conditions {
            let mut temp_matches = Vec::new();
            let confidence = self.evaluate_conditions(
                &[condition.clone()],
                events,
                current_event,
                &mut temp_matches,
            )?;

            if confidence == 0.0 {
                return Ok(0.0); // All conditions must match
            }

            total_confidence += confidence;
            all_matches.extend(temp_matches);
        }

        matched_events.extend(all_matches);
        Ok(total_confidence / conditions.len() as f64)
    }

    /// Get event property as string
    fn get_event_property(&self, event: &StreamEvent, property: &str) -> Option<String> {
        match event {
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
            | StreamEvent::QueryResultAdded { metadata, .. }
            | StreamEvent::QueryResultRemoved { metadata, .. }
            | StreamEvent::QueryCompleted { metadata, .. } => {
                metadata.properties.get(property).cloned()
            }
            StreamEvent::Heartbeat { .. } => None,
            // For events without metadata, return None
            _ => None,
        }
    }

    /// Get event property as numeric value
    fn get_event_numeric_property(&self, event: &StreamEvent, property: &str) -> Option<f64> {
        self.get_event_property(event, property)
            .and_then(|v| v.parse().ok())
    }

    /// Clear old matches based on retention policy
    pub fn cleanup_old_matches(&mut self, retention_duration: ChronoDuration) {
        let cutoff_time = Utc::now() - retention_duration;
        self.matches.retain(|m| m.match_time >= cutoff_time);
    }

    /// Get pattern statistics
    pub fn get_pattern_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        for match_item in &self.matches {
            *stats.entry(match_item.pattern_name.clone()).or_insert(0) += 1;
        }
        stats
    }
}

impl Default for ComplexEventProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced Complex Event Processing capabilities with ML and real-time analytics
/// 
/// Machine Learning-Enhanced Anomaly Detection for automatic pattern discovery
#[derive(Debug)]
pub struct AnomalyDetector {
    /// Statistical models for different event types
    baseline_models: HashMap<String, StatisticalModel>,
    /// Threshold for anomaly detection
    anomaly_threshold: f64,
    /// Learning rate for model updates
    learning_rate: f64,
    /// Historical data for model training
    historical_data: VecDeque<AnomalyDataPoint>,
    /// Detected anomalies
    detected_anomalies: Vec<AnomalyEvent>,
    /// Configuration for anomaly detection
    config: AnomalyDetectionConfig,
}

/// Statistical model for baseline behavior learning
#[derive(Debug, Clone)]
pub struct StatisticalModel {
    /// Mean values for numerical features
    means: HashMap<String, f64>,
    /// Standard deviations for numerical features
    std_devs: HashMap<String, f64>,
    /// Frequency distributions for categorical features
    frequencies: HashMap<String, HashMap<String, f64>>,
    /// Temporal patterns (hourly, daily, weekly)
    temporal_patterns: TemporalPattern,
    /// Number of samples used for training
    sample_count: usize,
    /// Last update timestamp
    last_updated: DateTime<Utc>,
}

/// Temporal pattern detection for time-based anomalies
#[derive(Debug, Clone)]
pub struct TemporalPattern {
    /// Hourly event frequency patterns
    hourly_patterns: [f64; 24],
    /// Daily event frequency patterns
    daily_patterns: [f64; 7],
    /// Seasonal trend coefficients
    seasonal_coefficients: Vec<f64>,
    /// Trend direction and strength
    trend_slope: f64,
}

/// Data point for anomaly detection training
#[derive(Debug, Clone)]
pub struct AnomalyDataPoint {
    /// Event features for analysis
    features: HashMap<String, f64>,
    /// Categorical attributes
    categorical_features: HashMap<String, String>,
    /// Timestamp for temporal analysis
    timestamp: DateTime<Utc>,
    /// Event source/type
    source: String,
}

/// Detected anomaly event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyEvent {
    /// Unique anomaly identifier
    pub id: String,
    /// Anomaly score (0.0 to 1.0)
    pub score: f64,
    /// Detected event that triggered anomaly
    pub event: StreamEvent,
    /// Type of anomaly detected
    pub anomaly_type: AnomalyType,
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,
    /// Explanation of why it's anomalous
    pub explanation: String,
    /// Related events in context
    pub context_events: Vec<StreamEvent>,
}

/// Types of anomalies that can be detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Statistical outlier in numerical values
    StatisticalOutlier,
    /// Unusual temporal pattern
    TemporalAnomaly,
    /// Rare event type or combination
    RareEvent,
    /// Unusual event sequence
    SequenceAnomaly,
    /// Volume-based anomaly (too many/few events)
    VolumeAnomaly,
    /// Complex multi-dimensional anomaly
    MultiDimensional,
}

/// Configuration for anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionConfig {
    /// Enable statistical anomaly detection
    pub enable_statistical: bool,
    /// Enable temporal anomaly detection
    pub enable_temporal: bool,
    /// Enable sequence anomaly detection
    pub enable_sequence: bool,
    /// Minimum anomaly score to report
    pub min_score_threshold: f64,
    /// Maximum historical data points to keep
    pub max_historical_samples: usize,
    /// Model update frequency
    pub update_frequency: Duration,
    /// Context window size for related events
    pub context_window_size: usize,
    /// Learning rate for adaptive models
    pub learning_rate: f64,
}

impl Default for AnomalyDetectionConfig {
    fn default() -> Self {
        Self {
            enable_statistical: true,
            enable_temporal: true,
            enable_sequence: true,
            min_score_threshold: 0.7,
            max_historical_samples: 10000,
            update_frequency: Duration::from_secs(300), // 5 minutes
            context_window_size: 10,
            learning_rate: 0.01,
        }
    }
}

/// Advanced temporal pattern detection for complex event relationships
#[derive(Debug)]
pub struct AdvancedTemporalProcessor {
    /// Advanced pattern definitions
    advanced_patterns: Vec<AdvancedTemporalPattern>,
    /// Causality analyzer for event relationships
    causality_analyzer: CausalityAnalyzer,
    /// Temporal state manager
    state_manager: TemporalStateManager,
    /// Trend analyzer for pattern evolution
    trend_analyzer: TrendAnalyzer,
    /// Configuration for temporal processing
    config: TemporalProcessingConfig,
}

/// Advanced temporal pattern with sophisticated relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedTemporalPattern {
    /// Pattern identifier
    pub id: String,
    /// Pattern name
    pub name: String,
    /// Advanced pattern conditions
    pub conditions: Vec<AdvancedPatternCondition>,
    /// Temporal constraints
    pub temporal_constraints: TemporalConstraints,
    /// Confidence threshold
    pub confidence_threshold: f64,
    /// Actions to execute when pattern matches
    pub actions: Vec<PatternAction>,
}

/// Advanced pattern conditions for sophisticated CEP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdvancedPatternCondition {
    /// Event A followed by Event B within time window
    FollowedBy {
        first: Box<PatternCondition>,
        second: Box<PatternCondition>,
        within: ChronoDuration,
        exactly: bool, // true = immediately followed, false = eventually followed
    },
    /// Event A not followed by Event B within time window
    NotFollowedBy {
        first: Box<PatternCondition>,
        second: Box<PatternCondition>,
        within: ChronoDuration,
    },
    /// Event occurs during another event's lifetime
    During {
        contained: Box<PatternCondition>,
        container_start: Box<PatternCondition>,
        container_end: Box<PatternCondition>,
    },
    /// Frequency-based pattern detection
    FrequencyThreshold {
        condition: Box<PatternCondition>,
        count: u32,
        within: ChronoDuration,
        comparison: FrequencyComparison,
    },
    /// Trend pattern in numerical values
    TrendPattern {
        field: String,
        trend_type: TrendType,
        sensitivity: f64,
        window: ChronoDuration,
    },
    /// Causal relationship between events
    CausalityPattern {
        cause: Box<PatternCondition>,
        effect: Box<PatternCondition>,
        max_delay: ChronoDuration,
        confidence: f64,
    },
    /// Statistical pattern (mean, deviation, etc.)
    StatisticalPattern {
        field: String,
        statistic: StatisticType,
        operator: ComparisonOperator,
        threshold: f64,
        window: ChronoDuration,
    },
}

/// Temporal constraints for pattern matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConstraints {
    /// Overall pattern timeout
    pub timeout: ChronoDuration,
    /// Maximum allowed gap between related events
    pub max_gap: Option<ChronoDuration>,
    /// Minimum required gap between related events
    pub min_gap: Option<ChronoDuration>,
    /// Temporal ordering requirement
    pub ordering: TemporalOrdering,
}

/// Frequency comparison operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FrequencyComparison {
    Exactly,
    AtLeast,
    AtMost,
    Between(u32, u32),
}

/// Trend types for pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendType {
    Increasing,
    Decreasing,
    Oscillating,
    Stable,
    Exponential,
    Linear,
}

/// Statistical types for pattern detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticType {
    Mean,
    Median,
    StandardDeviation,
    Variance,
    Percentile(u8),
    Range,
}

/// Temporal ordering requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalOrdering {
    Strict,      // Events must occur in exact order
    Partial,     // Some events can be reordered
    Unordered,   // Order doesn't matter
}

/// Causality analyzer for event relationships
#[derive(Debug)]
pub struct CausalityAnalyzer {
    /// Event correlation matrix
    correlation_matrix: HashMap<(String, String), CorrelationStrength>,
    /// Causal relationship discovery
    causal_relationships: Vec<CausalRelationship>,
    /// Statistical significance threshold
    significance_threshold: f64,
    /// Time window for causality analysis
    analysis_window: ChronoDuration,
}

/// Correlation strength between events
#[derive(Debug, Clone)]
pub struct CorrelationStrength {
    /// Pearson correlation coefficient
    coefficient: f64,
    /// Statistical significance
    p_value: f64,
    /// Sample size
    sample_size: usize,
    /// Last updated
    updated_at: DateTime<Utc>,
}

/// Discovered causal relationship
#[derive(Debug, Clone)]
pub struct CausalRelationship {
    /// Cause event type
    cause: String,
    /// Effect event type
    effect: String,
    /// Causal strength (0.0 to 1.0)
    strength: f64,
    /// Average delay between cause and effect
    average_delay: ChronoDuration,
    /// Confidence in the relationship
    confidence: f64,
}

/// Temporal state manager for stateful pattern detection
#[derive(Debug)]
pub struct TemporalStateManager {
    /// Active pattern states
    active_states: HashMap<String, PatternState>,
    /// State transition history
    transition_history: VecDeque<StateTransition>,
    /// State cleanup policy
    cleanup_policy: StateCleanupPolicy,
}

/// Pattern matching state
#[derive(Debug, Clone)]
pub struct PatternState {
    /// Pattern identifier
    pattern_id: String,
    /// Current state in pattern matching
    current_state: String,
    /// Matched events so far
    matched_events: Vec<StreamEvent>,
    /// State variables
    variables: HashMap<String, serde_json::Value>,
    /// Start timestamp
    started_at: DateTime<Utc>,
    /// Last update timestamp
    updated_at: DateTime<Utc>,
}

/// State transition record
#[derive(Debug, Clone)]
pub struct StateTransition {
    /// Pattern identifier
    pattern_id: String,
    /// Previous state
    from_state: String,
    /// New state
    to_state: String,
    /// Triggering event
    trigger_event: StreamEvent,
    /// Transition timestamp
    timestamp: DateTime<Utc>,
}

/// State cleanup policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateCleanupPolicy {
    /// Maximum state age before cleanup
    max_age: ChronoDuration,
    /// Maximum number of states to keep
    max_states: usize,
    /// Cleanup frequency
    cleanup_interval: Duration,
}

/// Real-time analytics engine for stream intelligence
#[derive(Debug)]
pub struct RealTimeAnalyticsEngine {
    /// KPI calculators
    kpi_calculators: HashMap<String, KpiCalculator>,
    /// Trend analyzers
    trend_analyzers: HashMap<String, TrendAnalyzer>,
    /// Forecasting models
    forecasting_models: HashMap<String, ForecastingModel>,
    /// Real-time metrics
    realtime_metrics: RealTimeMetrics,
    /// Analytics configuration
    config: AnalyticsConfig,
}

/// KPI (Key Performance Indicator) calculator
#[derive(Debug)]
pub struct KpiCalculator {
    /// KPI definition
    definition: KpiDefinition,
    /// Current value
    current_value: f64,
    /// Historical values
    history: VecDeque<KpiDataPoint>,
    /// Aggregation state
    aggregation_state: KpiAggregationState,
}

/// KPI definition and calculation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KpiDefinition {
    /// KPI name
    pub name: String,
    /// KPI type
    pub kpi_type: KpiType,
    /// Target field for calculation
    pub target_field: String,
    /// Calculation window
    pub window: ChronoDuration,
    /// Aggregation method
    pub aggregation: AggregateFunction,
    /// Alert thresholds
    pub thresholds: Vec<KpiThreshold>,
}

/// Types of KPIs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KpiType {
    /// Simple metric calculation
    Simple,
    /// Ratio between two metrics
    Ratio { numerator: String, denominator: String },
    /// Percentage calculation
    Percentage { part: String, whole: String },
    /// Growth rate calculation
    GrowthRate { period: ChronoDuration },
    /// Custom formula
    Custom { formula: String },
}

/// KPI threshold for alerting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KpiThreshold {
    /// Threshold name
    pub name: String,
    /// Threshold value
    pub value: f64,
    /// Comparison operator
    pub operator: ComparisonOperator,
    /// Alert severity
    pub severity: AlertSeverity,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// KPI data point for historical tracking
#[derive(Debug, Clone)]
pub struct KpiDataPoint {
    /// KPI value
    value: f64,
    /// Timestamp
    timestamp: DateTime<Utc>,
    /// Contributing events count
    event_count: usize,
}

/// Trend analyzer for pattern evolution
#[derive(Debug)]
pub struct TrendAnalyzer {
    /// Time series data
    time_series: VecDeque<TimeSeriesPoint>,
    /// Trend detection algorithm
    algorithm: TrendAlgorithm,
    /// Current trend
    current_trend: Option<DetectedTrend>,
    /// Trend sensitivity
    sensitivity: f64,
}

/// Time series data point
#[derive(Debug, Clone)]
pub struct TimeSeriesPoint {
    /// Timestamp
    timestamp: DateTime<Utc>,
    /// Value
    value: f64,
    /// Optional metadata
    metadata: HashMap<String, String>,
}

/// Trend detection algorithms
#[derive(Debug, Clone)]
pub enum TrendAlgorithm {
    SimpleMovingAverage { window: usize },
    ExponentialSmoothing { alpha: f64 },
    LinearRegression { window: usize },
    SeasonalDecomposition { period: usize },
}

/// Detected trend information
#[derive(Debug, Clone)]
pub struct DetectedTrend {
    /// Trend direction
    direction: TrendDirection,
    /// Trend strength (0.0 to 1.0)
    strength: f64,
    /// Confidence in trend detection
    confidence: f64,
    /// Trend start time
    start_time: DateTime<Utc>,
    /// Expected duration
    expected_duration: Option<ChronoDuration>,
}

/// Trend direction
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Oscillating,
    Unknown,
}

/// Forecasting model for predictive analytics
#[derive(Debug)]
pub struct ForecastingModel {
    /// Model type
    model_type: ForecastingModelType,
    /// Training data
    training_data: VecDeque<f64>,
    /// Model parameters
    parameters: HashMap<String, f64>,
    /// Forecast accuracy metrics
    accuracy: ForecastAccuracy,
}

/// Types of forecasting models
#[derive(Debug, Clone)]
pub enum ForecastingModelType {
    ARIMA { p: usize, d: usize, q: usize },
    ExponentialSmoothing { alpha: f64, beta: f64, gamma: f64 },
    LinearRegression,
    SeasonalNaive { season_length: usize },
    MovingAverage { window: usize },
}

/// Forecast accuracy metrics
#[derive(Debug, Clone)]
pub struct ForecastAccuracy {
    /// Mean Absolute Error
    mae: f64,
    /// Mean Squared Error
    mse: f64,
    /// Mean Absolute Percentage Error
    mape: f64,
    /// R-squared
    r_squared: f64,
}

/// Real-time metrics collection
#[derive(Debug)]
pub struct RealTimeMetrics {
    /// Current metrics values
    current_metrics: HashMap<String, f64>,
    /// Metrics history for trending
    metrics_history: HashMap<String, VecDeque<MetricPoint>>,
    /// Metric definitions
    metric_definitions: HashMap<String, MetricDefinition>,
}

/// Metric data point
#[derive(Debug, Clone)]
pub struct MetricPoint {
    /// Metric value
    value: f64,
    /// Timestamp
    timestamp: DateTime<Utc>,
    /// Tags for categorization
    tags: HashMap<String, String>,
}

/// Metric definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDefinition {
    /// Metric name
    pub name: String,
    /// Metric type
    pub metric_type: MetricType,
    /// Unit of measurement
    pub unit: String,
    /// Description
    pub description: String,
    /// Aggregation method
    pub aggregation: AggregateFunction,
}

/// Types of metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
    Timer,
}

/// Configuration for analytics engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    /// Enable KPI calculation
    pub enable_kpi: bool,
    /// Enable trend analysis
    pub enable_trend_analysis: bool,
    /// Enable forecasting
    pub enable_forecasting: bool,
    /// Maximum history retention
    pub max_history_retention: ChronoDuration,
    /// Update frequency for calculations
    pub update_frequency: Duration,
    /// Default forecast horizon
    pub default_forecast_horizon: ChronoDuration,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            enable_kpi: true,
            enable_trend_analysis: true,
            enable_forecasting: true,
            max_history_retention: ChronoDuration::days(30),
            update_frequency: Duration::from_secs(60),
            default_forecast_horizon: ChronoDuration::hours(24),
        }
    }
}

/// Configuration for temporal processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalProcessingConfig {
    /// Enable advanced pattern matching
    pub enable_advanced_patterns: bool,
    /// Enable causality analysis
    pub enable_causality_analysis: bool,
    /// Enable trend detection
    pub enable_trend_detection: bool,
    /// Maximum pattern complexity
    pub max_pattern_complexity: usize,
    /// State cleanup frequency
    pub state_cleanup_frequency: Duration,
    /// Causality analysis window
    pub causality_window: ChronoDuration,
}

impl Default for TemporalProcessingConfig {
    fn default() -> Self {
        Self {
            enable_advanced_patterns: true,
            enable_causality_analysis: true,
            enable_trend_detection: true,
            max_pattern_complexity: 10,
            state_cleanup_frequency: Duration::from_secs(300),
            causality_window: ChronoDuration::hours(24),
        }
    }
}

// Implementation methods for the advanced CEP components

impl AnomalyDetector {
    /// Create a new anomaly detector
    pub fn new(config: AnomalyDetectionConfig) -> Self {
        Self {
            baseline_models: HashMap::new(),
            anomaly_threshold: config.min_score_threshold,
            learning_rate: config.learning_rate,
            historical_data: VecDeque::with_capacity(config.max_historical_samples),
            detected_anomalies: Vec::new(),
            config,
        }
    }

    /// Process an event for anomaly detection
    pub async fn process_event(&mut self, event: &StreamEvent) -> Result<Option<AnomalyEvent>> {
        // Extract features from the event
        let data_point = self.extract_features(event)?;
        
        // Update or create baseline model
        self.update_baseline_model(&data_point).await?;
        
        // Calculate anomaly score
        let score = self.calculate_anomaly_score(&data_point)?;
        
        // Check if score exceeds threshold
        if score >= self.config.min_score_threshold {
            let anomaly = AnomalyEvent {
                id: Uuid::new_v4().to_string(),
                score,
                event: event.clone(),
                anomaly_type: self.classify_anomaly_type(&data_point, score),
                detected_at: Utc::now(),
                explanation: self.generate_explanation(&data_point, score),
                context_events: self.get_context_events(event).await,
            };
            
            self.detected_anomalies.push(anomaly.clone());
            Ok(Some(anomaly))
        } else {
            Ok(None)
        }
    }

    /// Extract numerical and categorical features from event
    fn extract_features(&self, event: &StreamEvent) -> Result<AnomalyDataPoint> {
        let mut features = HashMap::new();
        let mut categorical_features = HashMap::new();
        
        // Extract basic features
        features.insert("timestamp_hour".to_string(), event.timestamp().hour() as f64);
        features.insert("timestamp_day_of_week".to_string(), event.timestamp().weekday().num_days_from_monday() as f64);
        
        // Extract event-specific features
        categorical_features.insert("event_type".to_string(), format!("{:?}", event.event_type()));
        
        let metadata = event.metadata();
        // Note: EventMetadata doesn't have priority field, using defaults
        features.insert("priority".to_string(), 1.0); // Default priority
        
        // Use event_id length as a size proxy
        features.insert("size".to_string(), metadata.event_id.len() as f64);

        Ok(AnomalyDataPoint {
            features,
            categorical_features,
            timestamp: event.timestamp(),
            source: event.metadata().source.clone(),
        })
    }

    /// Update baseline statistical model
    async fn update_baseline_model(&mut self, data_point: &AnomalyDataPoint) -> Result<()> {
        let model = self.baseline_models
            .entry(data_point.source.clone())
            .or_insert_with(|| StatisticalModel {
                means: HashMap::new(),
                std_devs: HashMap::new(),
                frequencies: HashMap::new(),
                temporal_patterns: TemporalPattern {
                    hourly_patterns: [0.0; 24],
                    daily_patterns: [0.0; 7],
                    seasonal_coefficients: Vec::new(),
                    trend_slope: 0.0,
                },
                sample_count: 0,
                last_updated: Utc::now(),
            });

        // Update numerical feature statistics
        for (feature_name, &value) in &data_point.features {
            let old_mean = model.means.get(feature_name).unwrap_or(&0.0);
            let old_std = model.std_devs.get(feature_name).unwrap_or(&1.0);
            
            // Incremental mean and standard deviation update
            let new_count = model.sample_count + 1;
            let new_mean = (old_mean * model.sample_count as f64 + value) / new_count as f64;
            
            model.means.insert(feature_name.clone(), new_mean);
            
            if model.sample_count > 1 {
                let new_variance = ((model.sample_count - 1) as f64 * old_std.powi(2) + 
                                   (value - new_mean).powi(2)) / (new_count - 1) as f64;
                model.std_devs.insert(feature_name.clone(), new_variance.sqrt());
            }
        }

        // Update categorical feature frequencies
        for (feature_name, value) in &data_point.categorical_features {
            let freq_map = model.frequencies.entry(feature_name.clone()).or_insert_with(HashMap::new);
            let current_count = freq_map.get(value).unwrap_or(&0.0);
            freq_map.insert(value.clone(), current_count + 1.0);
        }

        // Update temporal patterns
        let hour = data_point.timestamp.hour() as usize;
        let day = data_point.timestamp.weekday().num_days_from_monday() as usize;
        model.temporal_patterns.hourly_patterns[hour] += 1.0;
        model.temporal_patterns.daily_patterns[day] += 1.0;

        model.sample_count += 1;
        model.last_updated = Utc::now();

        Ok(())
    }

    /// Calculate anomaly score for a data point
    fn calculate_anomaly_score(&self, data_point: &AnomalyDataPoint) -> Result<f64> {
        let model = self.baseline_models.get(&data_point.source)
            .ok_or_else(|| anyhow!("No baseline model for source: {}", data_point.source))?;

        let mut total_score = 0.0;
        let mut score_count = 0;

        // Calculate statistical anomaly scores
        if self.config.enable_statistical {
            for (feature_name, &value) in &data_point.features {
                if let (Some(&mean), Some(&std_dev)) = (model.means.get(feature_name), model.std_devs.get(feature_name)) {
                    if std_dev > 0.0 {
                        let z_score = ((value - mean) / std_dev).abs();
                        let stat_score = 1.0 - (-z_score.powi(2) / 2.0).exp(); // Gaussian-based score
                        total_score += stat_score;
                        score_count += 1;
                    }
                }
            }
        }

        // Calculate temporal anomaly scores
        if self.config.enable_temporal {
            let hour = data_point.timestamp.hour() as usize;
            let day = data_point.timestamp.weekday().num_days_from_monday() as usize;
            
            let expected_hourly = model.temporal_patterns.hourly_patterns[hour] / model.sample_count as f64;
            let expected_daily = model.temporal_patterns.daily_patterns[day] / model.sample_count as f64;
            
            let temporal_score = 1.0 - (expected_hourly * expected_daily).min(1.0);
            total_score += temporal_score;
            score_count += 1;
        }

        // Calculate categorical anomaly scores
        for (feature_name, value) in &data_point.categorical_features {
            if let Some(freq_map) = model.frequencies.get(feature_name) {
                let frequency = freq_map.get(value).unwrap_or(&0.0);
                let relative_freq = frequency / model.sample_count as f64;
                let categorical_score = 1.0 - relative_freq;
                total_score += categorical_score;
                score_count += 1;
            }
        }

        Ok(if score_count > 0 { total_score / score_count as f64 } else { 0.0 })
    }

    /// Classify the type of anomaly detected
    fn classify_anomaly_type(&self, data_point: &AnomalyDataPoint, score: f64) -> AnomalyType {
        // Simple classification based on score and features
        if score > 0.9 {
            AnomalyType::MultiDimensional
        } else if data_point.categorical_features.values().any(|v| v.contains("rare")) {
            AnomalyType::RareEvent
        } else if data_point.features.values().any(|&v| v > 1000.0) { // arbitrary threshold
            AnomalyType::StatisticalOutlier
        } else {
            AnomalyType::TemporalAnomaly
        }
    }

    /// Generate human-readable explanation for anomaly
    fn generate_explanation(&self, data_point: &AnomalyDataPoint, score: f64) -> String {
        format!(
            "Anomaly detected with score {:.3} for source '{}' at {}. Unusual patterns in features: {:?}",
            score,
            data_point.source,
            data_point.timestamp.format("%Y-%m-%d %H:%M:%S"),
            data_point.features.keys().collect::<Vec<_>>()
        )
    }

    /// Get related events for context
    async fn get_context_events(&self, _event: &StreamEvent) -> Vec<StreamEvent> {
        // Placeholder: In real implementation, would query recent events
        Vec::new()
    }
}

impl AdvancedTemporalProcessor {
    /// Create a new advanced temporal processor
    pub fn new(config: TemporalProcessingConfig) -> Self {
        Self {
            advanced_patterns: Vec::new(),
            causality_analyzer: CausalityAnalyzer::new(config.causality_window),
            state_manager: TemporalStateManager::new(),
            trend_analyzer: TrendAnalyzer::new(TrendAlgorithm::LinearRegression { window: 100 }),
            config,
        }
    }

    /// Add an advanced temporal pattern
    pub fn add_pattern(&mut self, pattern: AdvancedTemporalPattern) {
        self.advanced_patterns.push(pattern);
    }

    /// Process event for advanced temporal patterns
    pub async fn process_event(&mut self, event: &StreamEvent) -> Result<Vec<PatternMatch>> {
        let mut matches = Vec::new();

        // Process each advanced pattern
        let patterns = self.advanced_patterns.clone();
        for pattern in &patterns {
            if let Some(pattern_match) = self.evaluate_advanced_pattern(pattern, event).await? {
                matches.push(pattern_match);
            }
        }

        // Update causality analysis
        if self.config.enable_causality_analysis {
            self.causality_analyzer.process_event(event).await?;
        }

        // Update trend analysis
        if self.config.enable_trend_detection {
            self.trend_analyzer.process_event(event).await?;
        }

        Ok(matches)
    }

    /// Evaluate an advanced temporal pattern against an event
    async fn evaluate_advanced_pattern(
        &mut self,
        pattern: &AdvancedTemporalPattern,
        event: &StreamEvent,
    ) -> Result<Option<PatternMatch>> {
        // Get or create pattern state
        let state = self.state_manager.get_or_create_state(&pattern.id, event).await?;

        // Evaluate each condition
        for condition in &pattern.conditions {
            match self.evaluate_advanced_condition(condition, event, &state).await? {
                ConditionResult::Matched => {
                    // Update state and continue
                    self.state_manager.update_state(&pattern.id, event).await?;
                }
                ConditionResult::Failed => {
                    // Reset state
                    self.state_manager.reset_state(&pattern.id).await?;
                    return Ok(None);
                }
                ConditionResult::Pending => {
                    // Continue monitoring
                    return Ok(None);
                }
            }
        }

        // All conditions matched - pattern completed
        Ok(Some(PatternMatch {
            pattern_name: pattern.name.clone(),
            matched_events: state.matched_events.clone(),
            confidence: self.calculate_pattern_confidence(pattern, &state.matched_events),
            match_time: Utc::now(),
        }))
    }

    /// Evaluate an advanced pattern condition
    async fn evaluate_advanced_condition(
        &self,
        condition: &AdvancedPatternCondition,
        event: &StreamEvent,
        state: &PatternState,
    ) -> Result<ConditionResult> {
        match condition {
            AdvancedPatternCondition::FollowedBy { first, second, within, exactly } => {
                // Check if we've seen the first condition and now seeing the second
                let first_match = self.check_condition_in_history(first, &state.matched_events)?;
                if first_match {
                    // Check if current event matches second condition
                    if self.evaluate_basic_condition(second, event)? {
                        // Check timing constraint
                        if let Some(last_event) = state.matched_events.last() {
                            let time_diff = event.timestamp() - last_event.timestamp();
                            if time_diff <= *within {
                                return Ok(ConditionResult::Matched);
                            }
                        }
                    }
                }
                Ok(ConditionResult::Pending)
            }
            AdvancedPatternCondition::NotFollowedBy { first, second, within } => {
                // Check if first condition was matched but second hasn't occurred within time window
                let first_match = self.check_condition_in_history(first, &state.matched_events)?;
                if first_match {
                    if let Some(first_time) = state.matched_events.first().map(|e| e.timestamp()) {
                        let elapsed = event.timestamp() - first_time;
                        if elapsed > *within {
                            // Time window passed without second condition - pattern matched
                            return Ok(ConditionResult::Matched);
                        } else if self.evaluate_basic_condition(second, event)? {
                            // Second condition occurred within window - pattern failed
                            return Ok(ConditionResult::Failed);
                        }
                    }
                }
                Ok(ConditionResult::Pending)
            }
            AdvancedPatternCondition::FrequencyThreshold { condition, count, within, comparison } => {
                // Count matching events within time window
                let now = event.timestamp();
                let window_start = now - *within;
                
                let matching_count = state.matched_events.iter()
                    .filter(|e| e.timestamp() >= window_start && e.timestamp() <= now)
                    .filter(|e| self.evaluate_basic_condition(condition, e).unwrap_or(false))
                    .count() as u32;

                let threshold_met = match comparison {
                    FrequencyComparison::Exactly => matching_count == *count,
                    FrequencyComparison::AtLeast => matching_count >= *count,
                    FrequencyComparison::AtMost => matching_count <= *count,
                    FrequencyComparison::Between(min, max) => matching_count >= *min && matching_count <= *max,
                };

                if threshold_met {
                    Ok(ConditionResult::Matched)
                } else {
                    Ok(ConditionResult::Pending)
                }
            }
            AdvancedPatternCondition::TrendPattern { field, trend_type, sensitivity, window } => {
                // Analyze trend in the specified field over the time window
                let trend_result = self.analyze_field_trend(field, window, &state.matched_events, *sensitivity)?;
                
                let trend_matches = match trend_type {
                    TrendType::Increasing => trend_result.direction == TrendDirection::Increasing,
                    TrendType::Decreasing => trend_result.direction == TrendDirection::Decreasing,
                    TrendType::Stable => trend_result.direction == TrendDirection::Stable,
                    TrendType::Oscillating => trend_result.direction == TrendDirection::Oscillating,
                    _ => false, // More complex trend types would require additional analysis
                };

                if trend_matches && trend_result.confidence >= *sensitivity {
                    Ok(ConditionResult::Matched)
                } else {
                    Ok(ConditionResult::Pending)
                }
            }
            _ => {
                // For other advanced conditions, implement similar logic
                Ok(ConditionResult::Pending)
            }
        }
    }

    /// Check if a condition exists in event history
    fn check_condition_in_history(&self, condition: &PatternCondition, events: &[StreamEvent]) -> Result<bool> {
        for event in events {
            if self.evaluate_basic_condition(condition, event)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Evaluate a basic pattern condition (delegated to existing implementation)
    fn evaluate_basic_condition(&self, condition: &PatternCondition, event: &StreamEvent) -> Result<bool> {
        // This would delegate to the existing pattern condition evaluation logic
        // Placeholder implementation
        Ok(true)
    }

    /// Analyze trend in a specific field
    fn analyze_field_trend(
        &self,
        field: &str,
        window: &ChronoDuration,
        events: &[StreamEvent],
        sensitivity: f64,
    ) -> Result<DetectedTrend> {
        // Placeholder implementation for trend analysis
        Ok(DetectedTrend {
            direction: TrendDirection::Stable,
            strength: 0.5,
            confidence: sensitivity,
            start_time: Utc::now(),
            expected_duration: Some(*window),
        })
    }

    /// Calculate confidence score for pattern match
    fn calculate_pattern_confidence(&self, pattern: &AdvancedTemporalPattern, events: &[StreamEvent]) -> f64 {
        // Simple confidence calculation based on number of matched events
        let base_confidence = events.len() as f64 / pattern.conditions.len() as f64;
        base_confidence.min(1.0)
    }
}

/// Result of condition evaluation
#[derive(Debug, PartialEq)]
enum ConditionResult {
    Matched,
    Failed,
    Pending,
}

impl CausalityAnalyzer {
    pub fn new(analysis_window: ChronoDuration) -> Self {
        Self {
            correlation_matrix: HashMap::new(),
            causal_relationships: Vec::new(),
            significance_threshold: 0.05,
            analysis_window,
        }
    }

    pub async fn process_event(&mut self, event: &StreamEvent) -> Result<()> {
        // Placeholder implementation for causality analysis
        debug!("Processing event for causality analysis: {:?}", event.event_type());
        Ok(())
    }
}

impl TemporalStateManager {
    pub fn new() -> Self {
        Self {
            active_states: HashMap::new(),
            transition_history: VecDeque::with_capacity(1000),
            cleanup_policy: StateCleanupPolicy {
                max_age: ChronoDuration::hours(24),
                max_states: 1000,
                cleanup_interval: Duration::from_secs(300),
            },
        }
    }

    pub async fn get_or_create_state(&mut self, pattern_id: &str, event: &StreamEvent) -> Result<PatternState> {
        if let Some(state) = self.active_states.get(pattern_id) {
            Ok(state.clone())
        } else {
            let new_state = PatternState {
                pattern_id: pattern_id.to_string(),
                current_state: "initial".to_string(),
                matched_events: vec![event.clone()],
                variables: HashMap::new(),
                started_at: event.timestamp(),
                updated_at: event.timestamp(),
            };
            self.active_states.insert(pattern_id.to_string(), new_state.clone());
            Ok(new_state)
        }
    }

    pub async fn update_state(&mut self, pattern_id: &str, event: &StreamEvent) -> Result<()> {
        if let Some(state) = self.active_states.get_mut(pattern_id) {
            state.matched_events.push(event.clone());
            state.updated_at = event.timestamp();
        }
        Ok(())
    }

    pub async fn reset_state(&mut self, pattern_id: &str) -> Result<()> {
        self.active_states.remove(pattern_id);
        Ok(())
    }
}

impl TrendAnalyzer {
    pub fn new(algorithm: TrendAlgorithm) -> Self {
        Self {
            time_series: VecDeque::with_capacity(1000),
            algorithm,
            current_trend: None,
            sensitivity: 0.1,
        }
    }

    pub async fn process_event(&mut self, event: &StreamEvent) -> Result<()> {
        // Extract numerical value from event for trend analysis
        let value = self.extract_numerical_value(event)?;
        
        let point = TimeSeriesPoint {
            timestamp: event.timestamp(),
            value,
            metadata: HashMap::new(),
        };
        
        self.time_series.push_back(point);
        
        // Keep series bounded
        if self.time_series.len() > 1000 {
            self.time_series.pop_front();
        }
        
        // Update trend analysis
        self.update_trend_analysis()?;
        
        Ok(())
    }

    fn extract_numerical_value(&self, event: &StreamEvent) -> Result<f64> {
        // Placeholder: extract a numerical value from the event
        // In practice, this would be configurable
        Ok(event.timestamp().timestamp() as f64)
    }

    fn update_trend_analysis(&mut self) -> Result<()> {
        if self.time_series.len() < 10 {
            return Ok(()); // Not enough data for trend analysis
        }

        match &self.algorithm {
            TrendAlgorithm::LinearRegression { window } => {
                let window_size = (*window).min(self.time_series.len());
                let recent_data: Vec<f64> = self.time_series.iter()
                    .skip(self.time_series.len() - window_size)
                    .map(|p| p.value)
                    .collect();

                let trend = self.calculate_linear_trend(&recent_data)?;
                self.current_trend = Some(trend);
            }
            _ => {
                // Implement other trend algorithms as needed
            }
        }

        Ok(())
    }

    fn calculate_linear_trend(&self, data: &[f64]) -> Result<DetectedTrend> {
        if data.len() < 2 {
            return Ok(DetectedTrend {
                direction: TrendDirection::Unknown,
                strength: 0.0,
                confidence: 0.0,
                start_time: Utc::now(),
                expected_duration: None,
            });
        }

        // Simple linear regression slope calculation
        let n = data.len() as f64;
        let x_sum: f64 = (0..data.len()).map(|i| i as f64).sum();
        let y_sum: f64 = data.iter().sum();
        let xy_sum: f64 = data.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let x2_sum: f64 = (0..data.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum.powi(2));
        
        let direction = if slope > self.sensitivity {
            TrendDirection::Increasing
        } else if slope < -self.sensitivity {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        Ok(DetectedTrend {
            direction,
            strength: slope.abs(),
            confidence: 0.8, // Placeholder confidence calculation
            start_time: Utc::now(),
            expected_duration: Some(ChronoDuration::hours(1)),
        })
    }
}

impl RealTimeAnalyticsEngine {
    /// Create a new real-time analytics engine
    pub fn new(config: AnalyticsConfig) -> Self {
        Self {
            kpi_calculators: HashMap::new(),
            trend_analyzers: HashMap::new(),
            forecasting_models: HashMap::new(),
            realtime_metrics: RealTimeMetrics::new(),
            config,
        }
    }

    /// Add a KPI calculator
    pub fn add_kpi(&mut self, definition: KpiDefinition) {
        let calculator = KpiCalculator::new(definition.clone());
        self.kpi_calculators.insert(definition.name.clone(), calculator);
    }

    /// Process event for real-time analytics
    pub async fn process_event(&mut self, event: &StreamEvent) -> Result<AnalyticsResults> {
        let mut results = AnalyticsResults::default();

        // Update KPI calculations
        if self.config.enable_kpi {
            for (name, calculator) in &mut self.kpi_calculators {
                if let Some(kpi_result) = calculator.process_event(event).await? {
                    results.kpi_updates.insert(name.clone(), kpi_result);
                }
            }
        }

        // Update trend analysis
        if self.config.enable_trend_analysis {
            for (name, analyzer) in &mut self.trend_analyzers {
                if let Some(trend_update) = analyzer.process_event(event).await? {
                    results.trend_updates.insert(name.clone(), trend_update);
                }
            }
        }

        // Update real-time metrics
        self.realtime_metrics.update_from_event(event).await?;

        Ok(results)
    }
}

/// Results from analytics processing
#[derive(Debug, Default)]
pub struct AnalyticsResults {
    /// KPI calculation results
    pub kpi_updates: HashMap<String, KpiResult>,
    /// Trend analysis updates
    pub trend_updates: HashMap<String, TrendUpdate>,
    /// Generated alerts
    pub alerts: Vec<AnalyticsAlert>,
}

/// KPI calculation result
#[derive(Debug, Clone)]
pub struct KpiResult {
    /// KPI name
    pub name: String,
    /// Current value
    pub value: f64,
    /// Previous value for comparison
    pub previous_value: Option<f64>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Alert triggered
    pub alert: Option<AnalyticsAlert>,
}

/// Trend analysis update
#[derive(Debug, Clone)]
pub struct TrendUpdate {
    /// Metric name
    pub metric: String,
    /// Current trend
    pub trend: DetectedTrend,
    /// Forecast
    pub forecast: Option<ForecastResult>,
}

/// Analytics alert
#[derive(Debug, Clone)]
pub struct AnalyticsAlert {
    /// Alert identifier
    pub id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert message
    pub message: String,
    /// Severity level
    pub severity: AlertSeverity,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Metric or KPI that triggered the alert
    pub source: String,
    /// Current value
    pub value: f64,
    /// Threshold that was breached
    pub threshold: f64,
}

/// Types of analytics alerts
#[derive(Debug, Clone)]
pub enum AlertType {
    KpiThresholdBreached,
    TrendChangeDetected,
    AnomalyDetected,
    ForecastAlert,
}

/// Forecast result
#[derive(Debug, Clone)]
pub struct ForecastResult {
    /// Forecasted values
    pub values: Vec<f64>,
    /// Timestamps for forecasted values
    pub timestamps: Vec<DateTime<Utc>>,
    /// Confidence intervals
    pub confidence_intervals: Vec<(f64, f64)>,
    /// Forecast accuracy
    pub accuracy: f64,
}

impl KpiCalculator {
    pub fn new(definition: KpiDefinition) -> Self {
        Self {
            definition,
            current_value: 0.0,
            history: VecDeque::with_capacity(1000),
            aggregation_state: KpiAggregationState::new(),
        }
    }

    pub async fn process_event(&mut self, event: &StreamEvent) -> Result<Option<KpiResult>> {
        // Extract value from event based on KPI definition
        let value = self.extract_value_from_event(event)?;
        
        // Update aggregation state
        self.aggregation_state.update(value);
        
        // Calculate current KPI value
        let previous_value = self.current_value;
        self.current_value = match self.definition.aggregation {
            AggregateFunction::Count => self.aggregation_state.count as f64,
            AggregateFunction::Sum => self.aggregation_state.sum,
            AggregateFunction::Average => self.aggregation_state.sum / self.aggregation_state.count as f64,
            AggregateFunction::Min => self.aggregation_state.min,
            AggregateFunction::Max => self.aggregation_state.max,
            _ => value, // For other functions, use raw value
        };

        // Add to history
        let data_point = KpiDataPoint {
            value: self.current_value,
            timestamp: event.timestamp(),
            event_count: 1,
        };
        self.history.push_back(data_point);

        // Check for threshold alerts
        let alert = self.check_thresholds()?;

        Ok(Some(KpiResult {
            name: self.definition.name.clone(),
            value: self.current_value,
            previous_value: Some(previous_value),
            timestamp: event.timestamp(),
            alert,
        }))
    }

    fn extract_value_from_event(&self, event: &StreamEvent) -> Result<f64> {
        // Placeholder: Extract value based on target_field
        // In practice, this would parse the event data structure
        match self.definition.target_field.as_str() {
            "size" => Ok(event.metadata().event_id.len() as f64), // Use event_id length as size proxy
            "priority" => Ok(1.0), // Default priority since metadata doesn't have priority field
            _ => Ok(1.0), // Default to count
        }
    }

    fn check_thresholds(&self) -> Result<Option<AnalyticsAlert>> {
        for threshold in &self.definition.thresholds {
            let threshold_breached = match threshold.operator {
                ComparisonOperator::Equal => (self.current_value - threshold.value).abs() < f64::EPSILON,
                ComparisonOperator::GreaterThan => self.current_value > threshold.value,
                ComparisonOperator::LessThan => self.current_value < threshold.value,
                ComparisonOperator::GreaterOrEqual => self.current_value >= threshold.value,
                ComparisonOperator::LessOrEqual => self.current_value <= threshold.value,
                ComparisonOperator::NotEqual => (self.current_value - threshold.value).abs() > f64::EPSILON,
            };

            if threshold_breached {
                return Ok(Some(AnalyticsAlert {
                    id: Uuid::new_v4().to_string(),
                    alert_type: AlertType::KpiThresholdBreached,
                    message: format!("KPI '{}' breached threshold '{}': current value {:.2} {} {:.2}",
                                   self.definition.name, threshold.name, self.current_value,
                                   format!("{:?}", threshold.operator), threshold.value),
                    severity: threshold.severity.clone(),
                    timestamp: Utc::now(),
                    source: self.definition.name.clone(),
                    value: self.current_value,
                    threshold: threshold.value,
                }));
            }
        }

        Ok(None)
    }
}

/// Simple aggregation state for KPI calculations
#[derive(Debug)]
struct KpiAggregationState {
    count: usize,
    sum: f64,
    min: f64,
    max: f64,
}

impl KpiAggregationState {
    fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn update(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }
}

impl RealTimeMetrics {
    pub fn new() -> Self {
        Self {
            current_metrics: HashMap::new(),
            metrics_history: HashMap::new(),
            metric_definitions: HashMap::new(),
        }
    }

    pub async fn update_from_event(&mut self, event: &StreamEvent) -> Result<()> {
        // Update basic metrics from event
        self.update_metric("event_count", 1.0, event.timestamp()).await?;
        
        // Use event_id length as a size proxy since metadata doesn't have size field
        let metadata = event.metadata();
        self.update_metric("total_size", metadata.event_id.len() as f64, event.timestamp()).await?;

        Ok(())
    }

    async fn update_metric(&mut self, name: &str, value: f64, timestamp: DateTime<Utc>) -> Result<()> {
        // Update current value
        let current = self.current_metrics.entry(name.to_string()).or_insert(0.0);
        *current += value;

        // Add to history
        let history = self.metrics_history.entry(name.to_string()).or_insert_with(|| VecDeque::with_capacity(1000));
        history.push_back(MetricPoint {
            value,
            timestamp,
            tags: HashMap::new(),
        });

        // Keep history bounded
        if history.len() > 1000 {
            history.pop_front();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EventMetadata, StreamEvent};

    fn create_test_event(event_type: &str, timestamp: DateTime<Utc>) -> StreamEvent {
        StreamEvent::TripleAdded {
            subject: "http://example.org/subject".to_string(),
            predicate: "http://example.org/predicate".to_string(),
            object: "http://example.org/object".to_string(),
            graph: None,
            metadata: EventMetadata {
                event_id: Uuid::new_v4().to_string(),
                timestamp,
                source: event_type.to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: {
                    let mut props = HashMap::new();
                    props.insert("count".to_string(), "1".to_string());
                    props.insert("value".to_string(), "42.5".to_string());
                    props
                },
                checksum: None,
            },
        }
    }

    #[tokio::test]
    async fn test_tumbling_window() {
        let mut processor = EventProcessor::new();

        let window_config = WindowConfig {
            window_type: WindowType::Tumbling {
                duration: ChronoDuration::seconds(1),
            },
            aggregates: vec![AggregateFunction::Count],
            group_by: vec![],
            filter: None,
            allow_lateness: None,
            trigger: WindowTrigger::OnTime,
        };

        let window_id = processor.create_window(window_config);

        // Add some events
        let now = Utc::now();
        for i in 0..5 {
            let event = create_test_event("test", now + ChronoDuration::milliseconds(i * 100));
            let results = processor.process_event(event).await.unwrap();

            // Should not trigger until window ends
            if i < 4 {
                assert!(results.is_empty());
            }
        }

        // Force trigger
        let results = processor.trigger_all_windows().await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].event_count, 5);
    }

    #[tokio::test]
    async fn test_count_based_window() {
        let mut processor = EventProcessor::new();

        let window_config = WindowConfig {
            window_type: WindowType::CountBased { size: 3 },
            aggregates: vec![
                AggregateFunction::Count,
                AggregateFunction::Sum {
                    field: "value".to_string(),
                },
            ],
            group_by: vec![],
            filter: None,
            allow_lateness: None,
            trigger: WindowTrigger::OnCount(3),
        };

        let _window_id = processor.create_window(window_config);

        let now = Utc::now();
        let mut total_results = 0;

        for i in 0..7 {
            let event = create_test_event("test", now + ChronoDuration::seconds(i));
            let results = processor.process_event(event).await.unwrap();
            total_results += results.len();
        }

        // Should have triggered twice (at events 3 and 6)
        assert!(total_results >= 2);
    }

    #[tokio::test]
    async fn test_aggregation_functions() {
        let mut processor = EventProcessor::new();

        let window_config = WindowConfig {
            window_type: WindowType::CountBased { size: 5 },
            aggregates: vec![
                AggregateFunction::Count,
                AggregateFunction::Sum {
                    field: "value".to_string(),
                },
                AggregateFunction::Average {
                    field: "value".to_string(),
                },
                AggregateFunction::Min {
                    field: "value".to_string(),
                },
                AggregateFunction::Max {
                    field: "value".to_string(),
                },
                AggregateFunction::First,
                AggregateFunction::Last,
                AggregateFunction::Distinct {
                    field: "source".to_string(),
                },
            ],
            group_by: vec![],
            filter: None,
            allow_lateness: None,
            trigger: WindowTrigger::OnCount(5),
        };

        let _window_id = processor.create_window(window_config);

        let now = Utc::now();
        let mut all_results = Vec::new();
        for i in 0..5 {
            let event = create_test_event(
                &format!("source{}", i % 3),
                now + ChronoDuration::seconds(i),
            );
            let mut event_results = processor.process_event(event).await.unwrap();
            all_results.append(&mut event_results);
        }

        // The window should have auto-triggered on the 5th event
        assert_eq!(all_results.len(), 1);

        let result = &all_results[0];
        assert_eq!(result.event_count, 5);

        // Check aggregations
        assert!(result.aggregations.contains_key("Count"));
        assert!(result.aggregations.contains_key("Sum { field: \"value\" }"));
        assert!(result
            .aggregations
            .contains_key("Average { field: \"value\" }"));

        // Count should be 5
        if let Some(count_value) = result.aggregations.get("Count") {
            assert_eq!(count_value.as_u64(), Some(5));
        }
    }

    #[tokio::test]
    async fn test_sliding_window() {
        let mut processor = EventProcessor::new();

        let window_config = WindowConfig {
            window_type: WindowType::Sliding {
                duration: ChronoDuration::seconds(5),
                slide: ChronoDuration::seconds(1),
            },
            aggregates: vec![AggregateFunction::Count],
            group_by: vec![],
            filter: None,
            allow_lateness: None,
            trigger: WindowTrigger::OnTime,
        };

        let _window_id = processor.create_window(window_config);

        let now = Utc::now();
        for i in 0..10 {
            let event = create_test_event("test", now + ChronoDuration::seconds(i));
            processor.process_event(event).await.unwrap();
        }

        let stats = processor.get_stats();
        assert_eq!(stats.events_processed, 10);
    }

    #[tokio::test]
    async fn test_late_event_handling() {
        let mut processor = EventProcessor::new();

        let window_config = WindowConfig {
            window_type: WindowType::Tumbling {
                duration: ChronoDuration::seconds(1),
            },
            aggregates: vec![AggregateFunction::Count],
            group_by: vec![],
            filter: None,
            allow_lateness: Some(ChronoDuration::seconds(2)),
            trigger: WindowTrigger::OnTime,
        };

        let _window_id = processor.create_window(window_config);

        let now = Utc::now();

        // Add a normal event
        let event1 = create_test_event("test", now);
        processor.process_event(event1).await.unwrap();

        // Add a late event (from the past)
        let late_event = create_test_event("test", now - ChronoDuration::seconds(10));
        processor.process_event(late_event).await.unwrap();

        let stats = processor.get_stats();
        assert_eq!(stats.events_processed, 2);
        assert_eq!(stats.late_events, 1);
    }

    #[tokio::test]
    async fn test_window_creation_and_removal() {
        let mut processor = EventProcessor::new();

        let window_config = WindowConfig {
            window_type: WindowType::CountBased { size: 10 },
            aggregates: vec![AggregateFunction::Count],
            group_by: vec![],
            filter: None,
            allow_lateness: None,
            trigger: WindowTrigger::OnCount(10),
        };

        let window_id = processor.create_window(window_config);
        assert_eq!(processor.windows.len(), 1);

        processor.remove_window(&window_id).unwrap();
        assert_eq!(processor.windows.len(), 0);

        // Try to remove non-existent window
        let result = processor.remove_window("non-existent");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_hybrid_trigger() {
        let mut processor = EventProcessor::new();

        let window_config = WindowConfig {
            window_type: WindowType::CountBased { size: 100 },
            aggregates: vec![AggregateFunction::Count],
            group_by: vec![],
            filter: None,
            allow_lateness: None,
            trigger: WindowTrigger::Hybrid {
                time: ChronoDuration::milliseconds(100),
                count: 3,
            },
        };

        let _window_id = processor.create_window(window_config);

        let now = Utc::now();
        let mut triggered = false;

        // Add events - should trigger on count of 3
        for i in 0..3 {
            let event = create_test_event("test", now + ChronoDuration::milliseconds(i * 10));
            let results = processor.process_event(event).await.unwrap();
            if !results.is_empty() {
                triggered = true;
                break;
            }
        }

        assert!(triggered);
    }

    #[test]
    fn test_complex_event_processor() {
        let mut cep = ComplexEventProcessor::new();

        let pattern = EventPattern {
            name: "test_pattern".to_string(),
            conditions: vec![
                PatternCondition::EventType("triple_added".to_string()),
                PatternCondition::FieldEquals {
                    field: "source".to_string(),
                    value: "test".to_string(),
                },
            ],
            within: Some(ChronoDuration::seconds(10)),
            action: PatternAction::Log("Pattern matched".to_string()),
        };

        cep.add_pattern(pattern);
        assert_eq!(cep.patterns.len(), 1);
    }

    #[tokio::test]
    async fn test_processor_stats() {
        let mut processor = EventProcessor::new();

        let window_config = WindowConfig {
            window_type: WindowType::CountBased { size: 2 },
            aggregates: vec![AggregateFunction::Count],
            group_by: vec![],
            filter: None,
            allow_lateness: None,
            trigger: WindowTrigger::OnCount(2),
        };

        let _window_id = processor.create_window(window_config);

        let now = Utc::now();
        for i in 0..5 {
            let event = create_test_event("test", now + ChronoDuration::seconds(i));
            processor.process_event(event).await.unwrap();
        }

        let stats = processor.get_stats();
        assert_eq!(stats.events_processed, 5);
        assert!(stats.windows_created >= 1);
        assert!(stats.processing_latency_ms >= 0.0);
    }

    #[test]
    fn test_window_config_serialization() {
        let config = WindowConfig {
            window_type: WindowType::Tumbling {
                duration: ChronoDuration::seconds(60),
            },
            aggregates: vec![
                AggregateFunction::Count,
                AggregateFunction::Average {
                    field: "value".to_string(),
                },
            ],
            group_by: vec!["source".to_string()],
            filter: Some("event_type = 'triple_added'".to_string()),
            allow_lateness: Some(ChronoDuration::seconds(30)),
            trigger: WindowTrigger::OnTime,
        };

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: WindowConfig = serde_json::from_str(&serialized).unwrap();

        // Verify some key fields
        match deserialized.window_type {
            WindowType::Tumbling { duration } => {
                assert_eq!(duration, ChronoDuration::seconds(60));
            }
            _ => panic!("Wrong window type"),
        }

        assert_eq!(deserialized.aggregates.len(), 2);
        assert_eq!(deserialized.group_by.len(), 1);
    }
}
