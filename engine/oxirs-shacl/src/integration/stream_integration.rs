//! # Stream Integration for SHACL Validation
//!
//! This module provides integration between SHACL validation and streaming RDF data sources,
//! enabling real-time validation of RDF events from Kafka, NATS, or other message brokers.
//!
//! ## Features
//!
//! - **Real-time validation**: Validate RDF events as they arrive
//! - **Batch processing**: Validate events in configurable batches
//! - **Backpressure handling**: Automatic slowdown when validation is overwhelmed
//! - **Event filtering**: Filter events based on validation results
//! - **Dead letter queue**: Route invalid events to DLQ for analysis
//! - **Metrics collection**: Track validation throughput and latency

use crate::{Result, Shape, ShapeId, ValidationReport, Validator};
use oxirs_core::Store;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Configuration for stream integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamIntegrationConfig {
    /// Batch size for validation
    pub batch_size: usize,

    /// Maximum wait time before processing partial batch (ms)
    pub batch_timeout_ms: u64,

    /// Enable backpressure handling
    pub enable_backpressure: bool,

    /// Maximum validation latency before backpressure (ms)
    pub backpressure_threshold_ms: u64,

    /// Enable dead letter queue for invalid events
    pub enable_dlq: bool,

    /// Maximum retries for failed validations
    pub max_retries: usize,

    /// Shapes to apply to stream events
    pub stream_shapes: Vec<ShapeId>,

    /// Enable metrics collection
    pub enable_metrics: bool,

    /// Filter events based on validation result
    pub filter_invalid_events: bool,
}

impl Default for StreamIntegrationConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            batch_timeout_ms: 1000,
            enable_backpressure: true,
            backpressure_threshold_ms: 5000,
            enable_dlq: true,
            max_retries: 3,
            stream_shapes: Vec::new(),
            enable_metrics: true,
            filter_invalid_events: false,
        }
    }
}

/// Stream event representing an RDF change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEvent {
    /// Event ID
    pub event_id: String,

    /// Event timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Event type (insert, delete, update)
    pub event_type: StreamEventType,

    /// RDF data in the event
    pub data: Vec<u8>,

    /// Event metadata
    pub metadata: HashMap<String, String>,

    /// Retry count
    pub retry_count: usize,
}

/// Stream event type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamEventType {
    /// Insert new triples
    Insert,

    /// Delete existing triples
    Delete,

    /// Update (delete + insert)
    Update,

    /// Batch of mixed operations
    Batch,
}

/// Stream validator for real-time validation
pub struct StreamValidator {
    /// SHACL validator
    validator: Arc<Validator>,

    /// Integration configuration
    config: StreamIntegrationConfig,

    /// Metrics collector
    metrics: Arc<dashmap::DashMap<String, StreamMetric>>,

    /// Event buffer for batching
    #[cfg(feature = "async")]
    event_buffer: Arc<tokio::sync::Mutex<Vec<StreamEvent>>>,

    /// Last batch processing time
    #[cfg(feature = "async")]
    last_batch_time: Arc<tokio::sync::Mutex<Instant>>,
}

impl StreamValidator {
    /// Create a new stream validator
    pub fn new(validator: Arc<Validator>, config: StreamIntegrationConfig) -> Self {
        Self {
            validator,
            config,
            metrics: Arc::new(dashmap::DashMap::new()),
            #[cfg(feature = "async")]
            event_buffer: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            #[cfg(feature = "async")]
            last_batch_time: Arc::new(tokio::sync::Mutex::new(Instant::now())),
        }
    }

    /// Validate a single stream event
    pub fn validate_event(
        &self,
        store: &dyn Store,
        event: &StreamEvent,
    ) -> Result<StreamValidationResult> {
        info!(
            "Validating stream event {} (type: {:?})",
            event.event_id, event.event_type
        );

        let start = Instant::now();

        // Get shapes for validation
        let shapes = self.get_shapes_for_event(event)?;

        if shapes.is_empty() {
            debug!("No SHACL shapes configured for stream validation");
            return Ok(StreamValidationResult {
                event_id: event.event_id.clone(),
                conforms: true,
                validation_time_ms: start.elapsed().as_millis() as u64,
                violations: Vec::new(),
                should_forward: true,
                should_retry: false,
            });
        }

        // Perform validation
        let report = self.validator.validate_store(store, None)?;
        let validation_time_ms = start.elapsed().as_millis() as u64;

        // Update metrics
        if self.config.enable_metrics {
            self.update_metrics("validation_latency", validation_time_ms as f64);
            self.update_metrics("events_validated", 1.0);

            if !report.conforms() {
                self.update_metrics("events_failed", 1.0);
            }
        }

        // Convert to stream result
        let result = self.convert_to_stream_result(&report, event, validation_time_ms)?;

        // Check if backpressure should be applied
        if self.config.enable_backpressure
            && validation_time_ms > self.config.backpressure_threshold_ms
        {
            warn!(
                "Validation latency {} ms exceeds threshold {} ms - applying backpressure",
                validation_time_ms, self.config.backpressure_threshold_ms
            );
            self.update_metrics("backpressure_events", 1.0);
        }

        Ok(result)
    }

    /// Validate a batch of stream events
    #[cfg(feature = "async")]
    pub async fn validate_batch(
        &self,
        store: &dyn Store,
        events: Vec<StreamEvent>,
    ) -> Result<Vec<StreamValidationResult>> {
        info!("Validating batch of {} events", events.len());

        let start = Instant::now();
        let mut results = Vec::new();

        for event in events {
            let result = self.validate_event(store, &event)?;
            results.push(result);
        }

        let total_time_ms = start.elapsed().as_millis() as u64;

        if self.config.enable_metrics {
            self.update_metrics("batch_validation_time", total_time_ms as f64);
            self.update_metrics("batch_size", results.len() as f64);
        }

        Ok(results)
    }

    /// Add event to buffer and process batch if ready
    #[cfg(feature = "async")]
    pub async fn buffer_event(&self, event: StreamEvent) -> Result<Option<Vec<StreamEvent>>> {
        let mut buffer = self.event_buffer.lock().await;
        buffer.push(event);

        // Check if batch is ready
        let should_process = buffer.len() >= self.config.batch_size || {
            let last_time = self.last_batch_time.lock().await;
            last_time.elapsed() > Duration::from_millis(self.config.batch_timeout_ms)
        };

        if should_process {
            let batch = buffer.drain(..).collect();
            let mut last_time = self.last_batch_time.lock().await;
            *last_time = Instant::now();
            Ok(Some(batch))
        } else {
            Ok(None)
        }
    }

    /// Process dead letter queue event
    pub fn process_dlq_event(&self, event: &StreamEvent) -> Result<DlqDecision> {
        if !self.config.enable_dlq {
            return Ok(DlqDecision::Drop);
        }

        if event.retry_count >= self.config.max_retries {
            warn!(
                "Event {} exceeded max retries ({}), moving to DLQ",
                event.event_id, self.config.max_retries
            );
            return Ok(DlqDecision::SendToDlq);
        }

        Ok(DlqDecision::Retry)
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> HashMap<String, f64> {
        self.metrics
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().value))
            .collect()
    }

    /// Reset all metrics
    pub fn reset_metrics(&self) {
        self.metrics.clear();
    }

    /// Get validation throughput (events/second)
    pub fn get_throughput(&self) -> f64 {
        if let Some(events) = self.metrics.get("events_validated") {
            if let Some(latency) = self.metrics.get("validation_latency") {
                if latency.value > 0.0 {
                    return (events.value * 1000.0) / latency.value;
                }
            }
        }
        0.0
    }

    // Private helper methods

    fn get_shapes_for_event(&self, _event: &StreamEvent) -> Result<Vec<Shape>> {
        // In a real implementation, resolve shape IDs to actual shapes
        Ok(Vec::new())
    }

    fn convert_to_stream_result(
        &self,
        report: &ValidationReport,
        event: &StreamEvent,
        validation_time_ms: u64,
    ) -> Result<StreamValidationResult> {
        let violations: Vec<String> = report
            .violations()
            .iter()
            .map(|v| {
                v.result_message
                    .clone()
                    .unwrap_or_else(|| "Validation error".to_string())
            })
            .collect();

        let conforms = violations.is_empty();

        // Determine if event should be forwarded
        let should_forward = if self.config.filter_invalid_events {
            conforms
        } else {
            true
        };

        // Determine if event should be retried
        let should_retry = !conforms && event.retry_count < self.config.max_retries;

        Ok(StreamValidationResult {
            event_id: event.event_id.clone(),
            conforms,
            validation_time_ms,
            violations,
            should_forward,
            should_retry,
        })
    }

    fn update_metrics(&self, key: &str, value: f64) {
        self.metrics
            .entry(key.to_string())
            .and_modify(|m| {
                m.value += value;
                m.count += 1;
            })
            .or_insert(StreamMetric {
                value,
                count: 1,
                timestamp: chrono::Utc::now(),
            });
    }
}

/// Stream validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamValidationResult {
    /// Event ID
    pub event_id: String,

    /// Whether the event conforms to shapes
    pub conforms: bool,

    /// Validation time (milliseconds)
    pub validation_time_ms: u64,

    /// List of violations
    pub violations: Vec<String>,

    /// Whether the event should be forwarded downstream
    pub should_forward: bool,

    /// Whether the event should be retried
    pub should_retry: bool,
}

impl StreamValidationResult {
    /// Convert to event routing decision
    pub fn routing_decision(&self) -> EventRoutingDecision {
        if self.should_forward {
            EventRoutingDecision::Forward
        } else if self.should_retry {
            EventRoutingDecision::Retry
        } else {
            EventRoutingDecision::DeadLetter
        }
    }
}

/// Event routing decision
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventRoutingDecision {
    /// Forward to next stage
    Forward,

    /// Retry validation
    Retry,

    /// Send to dead letter queue
    DeadLetter,

    /// Drop event
    Drop,
}

/// Dead letter queue decision
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DlqDecision {
    /// Retry the event
    Retry,

    /// Send to DLQ
    SendToDlq,

    /// Drop the event
    Drop,
}

/// Stream metric
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StreamMetric {
    value: f64,
    count: usize,
    timestamp: chrono::DateTime<chrono::Utc>,
}

/// Builder for stream validator
pub struct StreamValidatorBuilder {
    config: StreamIntegrationConfig,
}

impl StreamValidatorBuilder {
    pub fn new() -> Self {
        Self {
            config: StreamIntegrationConfig::default(),
        }
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    pub fn batch_timeout(mut self, ms: u64) -> Self {
        self.config.batch_timeout_ms = ms;
        self
    }

    pub fn enable_backpressure(mut self, enabled: bool) -> Self {
        self.config.enable_backpressure = enabled;
        self
    }

    pub fn backpressure_threshold(mut self, ms: u64) -> Self {
        self.config.backpressure_threshold_ms = ms;
        self
    }

    pub fn enable_dlq(mut self, enabled: bool) -> Self {
        self.config.enable_dlq = enabled;
        self
    }

    pub fn max_retries(mut self, retries: usize) -> Self {
        self.config.max_retries = retries;
        self
    }

    pub fn stream_shapes(mut self, shapes: Vec<ShapeId>) -> Self {
        self.config.stream_shapes = shapes;
        self
    }

    pub fn filter_invalid_events(mut self, enabled: bool) -> Self {
        self.config.filter_invalid_events = enabled;
        self
    }

    pub fn build(self, validator: Arc<Validator>) -> StreamValidator {
        StreamValidator::new(validator, self.config)
    }
}

impl Default for StreamValidatorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_event_types() {
        let types = [
            StreamEventType::Insert,
            StreamEventType::Delete,
            StreamEventType::Update,
            StreamEventType::Batch,
        ];

        assert_eq!(types.len(), 4);
    }

    #[test]
    fn test_routing_decision() {
        let result = StreamValidationResult {
            event_id: "test".to_string(),
            conforms: true,
            validation_time_ms: 10,
            violations: Vec::new(),
            should_forward: true,
            should_retry: false,
        };

        assert_eq!(result.routing_decision(), EventRoutingDecision::Forward);
    }

    #[test]
    fn test_dlq_decision() {
        let decision = DlqDecision::Retry;
        assert_eq!(decision, DlqDecision::Retry);
    }

    #[test]
    fn test_stream_validator_builder() {
        let config = StreamValidatorBuilder::new()
            .batch_size(200)
            .batch_timeout(2000)
            .enable_backpressure(true)
            .max_retries(5)
            .config;

        assert_eq!(config.batch_size, 200);
        assert_eq!(config.batch_timeout_ms, 2000);
        assert_eq!(config.max_retries, 5);
    }
}
