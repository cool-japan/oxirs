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
use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
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
            | StreamEvent::ShapeUpdated { metadata, .. } => metadata.timestamp,
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
            let key = format!("{:?}", aggregate);

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
                    | StreamEvent::SparqlUpdate { metadata, .. }
                    | StreamEvent::TransactionBegin { metadata, .. }
                    | StreamEvent::TransactionCommit { metadata, .. }
                    | StreamEvent::TransactionAbort { metadata, .. }
                    | StreamEvent::SchemaChanged { metadata, .. } => {
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
                    | StreamEvent::SparqlUpdate { metadata, .. }
                    | StreamEvent::TransactionBegin { metadata, .. }
                    | StreamEvent::TransactionCommit { metadata, .. }
                    | StreamEvent::TransactionAbort { metadata, .. }
                    | StreamEvent::SchemaChanged { metadata, .. } => {
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
            | StreamEvent::ShapeUpdated { metadata, .. } => metadata.source.clone(),
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
                AggregationState::Count(count) => serde_json::Value::Number((*count).into()),
                AggregationState::Sum(sum) => serde_json::Value::Number(
                    serde_json::Number::from_f64(*sum).unwrap_or_else(|| 0.into()),
                ),
                AggregationState::Average { sum, count } => {
                    let avg = if *count > 0 {
                        sum / (*count as f64)
                    } else {
                        0.0
                    };
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(avg).unwrap_or_else(|| 0.into()),
                    )
                }
                AggregationState::Min(min) => serde_json::Value::Number(
                    serde_json::Number::from_f64(*min).unwrap_or_else(|| 0.into()),
                ),
                AggregationState::Max(max) => serde_json::Value::Number(
                    serde_json::Number::from_f64(*max).unwrap_or_else(|| 0.into()),
                ),
                AggregationState::First(_) => serde_json::Value::String("first_event".to_string()),
                AggregationState::Last(_) => serde_json::Value::String("last_event".to_string()),
                AggregationState::Distinct(set) => serde_json::Value::Number(set.len().into()),
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

#[derive(Debug, Clone)]
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
        for i in 0..5 {
            let event = create_test_event(
                &format!("source{}", i % 3),
                now + ChronoDuration::seconds(i),
            );
            processor.process_event(event).await.unwrap();
        }

        let results = processor.trigger_all_windows().await.unwrap();
        assert_eq!(results.len(), 1);

        let result = &results[0];
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
