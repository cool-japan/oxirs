//! # ML Model Integration for Stream Processing
//!
//! Provides ML inference capabilities embedded in the stream processing pipeline:
//!
//! - [`StreamingModelRunner`]: Runs ML inference on stream events with batching
//! - [`StreamAnomalyDetector`]: Z-score based streaming anomaly detection with sliding window
//! - [`StreamFeatureExtractor`]: Configurable feature extraction from RDF stream events

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use scirs2_core::ndarray_ext::Array1;

use crate::event::StreamEvent;

// ─── Model Configuration ─────────────────────────────────────────────────────

/// Configuration for a streaming model runner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path or identifier for the model
    pub model_path: String,
    /// Maximum batch size before forcing inference
    pub batch_size: usize,
    /// Maximum latency before forcing inference (even if batch is not full)
    pub max_latency_ms: u64,
    /// Number of input features expected
    pub input_features: usize,
    /// Model name for logging
    pub model_name: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: "default".to_string(),
            batch_size: 32,
            max_latency_ms: 100,
            input_features: 4,
            model_name: "default-model".to_string(),
        }
    }
}

/// A single prediction from the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    /// Predicted value or class
    pub value: f64,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Source event identifier
    pub source_event_id: String,
    /// Timestamp of the prediction
    pub predicted_at: DateTime<Utc>,
    /// Model that produced the prediction
    pub model_name: String,
}

/// Statistics for the streaming model runner
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelRunnerStats {
    /// Total events processed
    pub events_processed: u64,
    /// Total batches executed
    pub batches_executed: u64,
    /// Total predictions produced
    pub predictions_produced: u64,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Average inference time per batch (milliseconds)
    pub avg_inference_time_ms: f64,
    /// Batches triggered by size threshold
    pub size_triggered_batches: u64,
    /// Batches triggered by latency threshold
    pub latency_triggered_batches: u64,
}

/// A pending event waiting to be included in a batch
#[derive(Debug, Clone)]
struct PendingEvent {
    features: Array1<f64>,
    event_id: String,
    queued_at: Instant,
}

/// Model weights for a simple linear model (more models can be added)
#[derive(Debug, Clone)]
struct LinearModelWeights {
    weights: Array1<f64>,
    bias: f64,
}

/// Runs ML inference on stream events with automatic batching.
///
/// Events are collected until either `batch_size` events accumulate
/// or `max_latency_ms` elapses, then inference is run on the batch.
pub struct StreamingModelRunner {
    config: ModelConfig,
    /// Pending events waiting for batch inference
    pending: Arc<RwLock<Vec<PendingEvent>>>,
    /// Model weights (simple linear model for now)
    model: Arc<RwLock<LinearModelWeights>>,
    /// Runner statistics
    stats: Arc<RwLock<ModelRunnerStats>>,
    /// When the oldest pending event was queued
    batch_start: Arc<RwLock<Option<Instant>>>,
}

impl StreamingModelRunner {
    /// Creates a new streaming model runner.
    pub fn new(config: ModelConfig) -> Self {
        // Initialize with small default weights
        let weights = Array1::from_vec(vec![0.1; config.input_features]);
        Self {
            config: config.clone(),
            pending: Arc::new(RwLock::new(Vec::with_capacity(config.batch_size))),
            model: Arc::new(RwLock::new(LinearModelWeights { weights, bias: 0.0 })),
            stats: Arc::new(RwLock::new(ModelRunnerStats::default())),
            batch_start: Arc::new(RwLock::new(None)),
        }
    }

    /// Enqueues an event for prediction.
    ///
    /// Returns predictions if a batch was triggered.
    pub fn enqueue(&self, features: Array1<f64>, event_id: String) -> Option<Vec<Prediction>> {
        if features.len() != self.config.input_features {
            warn!(
                "Feature dimension mismatch: expected {}, got {}",
                self.config.input_features,
                features.len()
            );
            return None;
        }

        let mut pending = self.pending.write();
        if pending.is_empty() {
            *self.batch_start.write() = Some(Instant::now());
        }
        pending.push(PendingEvent {
            features,
            event_id,
            queued_at: Instant::now(),
        });

        // Check if batch should be triggered
        if pending.len() >= self.config.batch_size {
            let events: Vec<PendingEvent> = std::mem::take(&mut *pending);
            drop(pending);
            *self.batch_start.write() = None;
            self.stats.write().size_triggered_batches += 1;
            Some(self.run_inference(events))
        } else {
            None
        }
    }

    /// Flushes any pending events if the latency threshold has been exceeded.
    ///
    /// Returns predictions if flush was needed.
    pub fn flush_if_due(&self) -> Option<Vec<Prediction>> {
        let should_flush = {
            let batch_start = self.batch_start.read();
            match *batch_start {
                Some(start) => start.elapsed().as_millis() as u64 >= self.config.max_latency_ms,
                None => false,
            }
        };

        if should_flush {
            let mut pending = self.pending.write();
            if pending.is_empty() {
                return None;
            }
            let events: Vec<PendingEvent> = std::mem::take(&mut *pending);
            drop(pending);
            *self.batch_start.write() = None;
            self.stats.write().latency_triggered_batches += 1;
            Some(self.run_inference(events))
        } else {
            None
        }
    }

    /// Forces inference on all pending events regardless of thresholds.
    pub fn flush(&self) -> Vec<Prediction> {
        let mut pending = self.pending.write();
        if pending.is_empty() {
            return Vec::new();
        }
        let events: Vec<PendingEvent> = std::mem::take(&mut *pending);
        drop(pending);
        *self.batch_start.write() = None;
        self.run_inference(events)
    }

    /// Runs batched inference directly on a slice of stream events.
    pub fn predict(&self, events: &[(Array1<f64>, String)]) -> Vec<Prediction> {
        let pending_events: Vec<PendingEvent> = events
            .iter()
            .map(|(features, event_id)| PendingEvent {
                features: features.clone(),
                event_id: event_id.clone(),
                queued_at: Instant::now(),
            })
            .collect();
        self.run_inference(pending_events)
    }

    /// Updates the model weights.
    pub fn update_weights(&self, weights: Array1<f64>, bias: f64) {
        let mut model = self.model.write();
        model.weights = weights;
        model.bias = bias;
        info!("Model {} weights updated", self.config.model_name);
    }

    /// Returns runner statistics.
    pub fn stats(&self) -> ModelRunnerStats {
        self.stats.read().clone()
    }

    /// Returns the number of pending events.
    pub fn pending_count(&self) -> usize {
        self.pending.read().len()
    }

    /// Internal inference function.
    fn run_inference(&self, events: Vec<PendingEvent>) -> Vec<Prediction> {
        let start = Instant::now();
        let model = self.model.read();
        let batch_size = events.len();

        let predictions: Vec<Prediction> = events
            .iter()
            .map(|event| {
                let mut value = model.bias;
                let n = model.weights.len().min(event.features.len());
                for i in 0..n {
                    value += model.weights[i] * event.features[i];
                }
                // Sigmoid for confidence
                let confidence = 1.0 / (1.0 + (-value).exp());

                Prediction {
                    value,
                    confidence: confidence.clamp(0.0, 1.0),
                    source_event_id: event.event_id.clone(),
                    predicted_at: Utc::now(),
                    model_name: self.config.model_name.clone(),
                }
            })
            .collect();

        let elapsed_ms = start.elapsed().as_micros() as f64 / 1000.0;

        let mut stats = self.stats.write();
        stats.events_processed += batch_size as u64;
        stats.batches_executed += 1;
        stats.predictions_produced += predictions.len() as u64;
        let total_batches = stats.batches_executed as f64;
        stats.avg_batch_size =
            (stats.avg_batch_size * (total_batches - 1.0) + batch_size as f64) / total_batches;
        stats.avg_inference_time_ms =
            (stats.avg_inference_time_ms * (total_batches - 1.0) + elapsed_ms) / total_batches;

        debug!(
            "Inference batch: {} events, {:.2}ms",
            batch_size, elapsed_ms
        );

        predictions
    }
}

// ─── Streaming Anomaly Detector ──────────────────────────────────────────────

/// Configuration for the streaming anomaly detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectorConfig {
    /// Z-score threshold for anomaly detection
    pub sigma_threshold: f64,
    /// Sliding window size for statistics computation
    pub window_size: usize,
    /// Minimum samples before detection starts
    pub min_samples: usize,
    /// Adaptive threshold learning rate (0.0 = fixed, 1.0 = fully adaptive)
    pub adaptive_rate: f64,
}

impl Default for AnomalyDetectorConfig {
    fn default() -> Self {
        Self {
            sigma_threshold: 3.0,
            window_size: 100,
            min_samples: 10,
            adaptive_rate: 0.0,
        }
    }
}

/// Result of anomaly detection on a single value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyCheckResult {
    /// Whether the value is anomalous
    pub is_anomaly: bool,
    /// The Z-score of the value
    pub z_score: f64,
    /// The current mean of the sliding window
    pub window_mean: f64,
    /// The current standard deviation of the sliding window
    pub window_stddev: f64,
    /// The effective threshold used
    pub threshold: f64,
    /// Number of samples in the window
    pub window_samples: usize,
}

/// Statistics for the anomaly detector
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnomalyDetectorStats {
    /// Total values processed
    pub values_processed: u64,
    /// Total anomalies detected
    pub anomalies_detected: u64,
    /// Current window mean
    pub current_mean: f64,
    /// Current window stddev
    pub current_stddev: f64,
    /// Detection rate
    pub detection_rate: f64,
}

/// Z-score based streaming anomaly detector with a sliding window.
///
/// Maintains a sliding window of recent values, computes running mean and
/// standard deviation, and flags values whose Z-score exceeds the configured
/// sigma threshold.
pub struct StreamAnomalyDetector {
    config: AnomalyDetectorConfig,
    /// Sliding window of recent values
    window: Arc<RwLock<VecDeque<f64>>>,
    /// Running sum for incremental mean computation
    running_sum: Arc<RwLock<f64>>,
    /// Running sum of squares for incremental stddev
    running_sum_sq: Arc<RwLock<f64>>,
    /// Effective threshold (may be adapted over time)
    effective_threshold: Arc<RwLock<f64>>,
    /// Statistics
    stats: Arc<RwLock<AnomalyDetectorStats>>,
}

impl StreamAnomalyDetector {
    /// Creates a new anomaly detector.
    pub fn new(config: AnomalyDetectorConfig) -> Self {
        let threshold = config.sigma_threshold;
        Self {
            config: config.clone(),
            window: Arc::new(RwLock::new(VecDeque::with_capacity(config.window_size))),
            running_sum: Arc::new(RwLock::new(0.0)),
            running_sum_sq: Arc::new(RwLock::new(0.0)),
            effective_threshold: Arc::new(RwLock::new(threshold)),
            stats: Arc::new(RwLock::new(AnomalyDetectorStats::default())),
        }
    }

    /// Checks whether a value is anomalous.
    pub fn is_anomaly(&self, value: f64) -> AnomalyCheckResult {
        let mut window = self.window.write();
        let mut sum = self.running_sum.write();
        let mut sum_sq = self.running_sum_sq.write();

        // Add value to window
        if window.len() >= self.config.window_size {
            if let Some(removed) = window.pop_front() {
                *sum -= removed;
                *sum_sq -= removed * removed;
            }
        }
        window.push_back(value);
        *sum += value;
        *sum_sq += value * value;

        let n = window.len();

        let mut stats = self.stats.write();
        stats.values_processed += 1;

        // Need minimum samples
        if n < self.config.min_samples {
            return AnomalyCheckResult {
                is_anomaly: false,
                z_score: 0.0,
                window_mean: if n > 0 { *sum / n as f64 } else { 0.0 },
                window_stddev: 0.0,
                threshold: *self.effective_threshold.read(),
                window_samples: n,
            };
        }

        let mean = *sum / n as f64;
        let variance = (*sum_sq / n as f64) - (mean * mean);
        let stddev = if variance > 0.0 { variance.sqrt() } else { 0.0 };

        let z_score = if stddev > 1e-10 {
            (value - mean).abs() / stddev
        } else {
            0.0
        };

        let threshold = *self.effective_threshold.read();
        let is_anomaly = z_score > threshold;

        if is_anomaly {
            stats.anomalies_detected += 1;
        }
        stats.current_mean = mean;
        stats.current_stddev = stddev;
        stats.detection_rate = if stats.values_processed > 0 {
            stats.anomalies_detected as f64 / stats.values_processed as f64
        } else {
            0.0
        };

        AnomalyCheckResult {
            is_anomaly,
            z_score,
            window_mean: mean,
            window_stddev: stddev,
            threshold,
            window_samples: n,
        }
    }

    /// Provides feedback to adapt the threshold.
    pub fn feedback(&self, was_true_anomaly: bool) {
        if self.config.adaptive_rate <= 0.0 {
            return;
        }
        let mut threshold = self.effective_threshold.write();
        if was_true_anomaly {
            // Lower threshold slightly to catch more
            *threshold *= 1.0 - (self.config.adaptive_rate * 0.02);
        } else {
            // Raise threshold slightly to reduce false positives
            *threshold *= 1.0 + (self.config.adaptive_rate * 0.02);
        }
        // Clamp to reasonable range
        *threshold = threshold.clamp(1.0, 10.0);
    }

    /// Returns detector statistics.
    pub fn stats(&self) -> AnomalyDetectorStats {
        self.stats.read().clone()
    }

    /// Returns the current effective threshold.
    pub fn effective_threshold(&self) -> f64 {
        *self.effective_threshold.read()
    }

    /// Resets the detector state.
    pub fn reset(&self) {
        self.window.write().clear();
        *self.running_sum.write() = 0.0;
        *self.running_sum_sq.write() = 0.0;
        *self.stats.write() = AnomalyDetectorStats::default();
    }
}

// ─── Feature Extractor ───────────────────────────────────────────────────────

/// A feature definition describing how to extract a numeric feature from an event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDefinition {
    /// Feature name
    pub name: String,
    /// Predicate selector: if the event's predicate contains this string, extract
    pub predicate_selector: Option<String>,
    /// Aggregation type for window-based features
    pub aggregation: FeatureAggregation,
}

/// Aggregation type for a feature
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FeatureAggregation {
    /// Use the latest value
    Latest,
    /// Count occurrences in window
    Count,
    /// Sum values in window
    Sum,
    /// Compute mean over window
    Mean,
}

/// Configuration for the feature extractor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractorConfig {
    /// Feature definitions
    pub features: Vec<FeatureDefinition>,
    /// Window size for aggregation-based features
    pub window_size: usize,
}

impl Default for FeatureExtractorConfig {
    fn default() -> Self {
        Self {
            features: vec![
                FeatureDefinition {
                    name: "event_count".to_string(),
                    predicate_selector: None,
                    aggregation: FeatureAggregation::Count,
                },
                FeatureDefinition {
                    name: "event_rate".to_string(),
                    predicate_selector: None,
                    aggregation: FeatureAggregation::Mean,
                },
            ],
            window_size: 50,
        }
    }
}

/// Extracted feature vector
#[derive(Debug, Clone)]
pub struct ExtractedFeatures {
    /// Feature values as a numeric array
    pub values: Array1<f64>,
    /// Feature names (corresponding to values)
    pub names: Vec<String>,
    /// Timestamp of extraction
    pub extracted_at: DateTime<Utc>,
    /// Source event ID
    pub event_id: String,
}

/// Configurable feature extractor for RDF stream events.
///
/// Extracts numeric features from stream events based on configured
/// feature definitions with predicate selectors and aggregation types.
pub struct StreamFeatureExtractor {
    config: FeatureExtractorConfig,
    /// History window of events for aggregation features
    history: Arc<RwLock<VecDeque<EventSnapshot>>>,
    /// Per-feature running values for aggregation
    running_values: Arc<RwLock<HashMap<String, VecDeque<f64>>>>,
}

/// Lightweight snapshot of an event for windowed features
#[derive(Debug, Clone)]
struct EventSnapshot {
    event_type: String,
    predicate: Option<String>,
    timestamp: Instant,
}

impl StreamFeatureExtractor {
    /// Creates a new feature extractor.
    pub fn new(config: FeatureExtractorConfig) -> Self {
        Self {
            config: config.clone(),
            history: Arc::new(RwLock::new(VecDeque::with_capacity(config.window_size))),
            running_values: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Extracts features from a stream event.
    pub fn extract(&self, event: &StreamEvent, event_id: &str) -> ExtractedFeatures {
        let event_type = Self::event_type_name(event);
        let predicate = Self::extract_predicate(event);

        // Update history
        let mut history = self.history.write();
        history.push_back(EventSnapshot {
            event_type: event_type.clone(),
            predicate: predicate.clone(),
            timestamp: Instant::now(),
        });
        while history.len() > self.config.window_size {
            history.pop_front();
        }
        let history_len = history.len();

        // Compute features
        let mut values = Vec::with_capacity(self.config.features.len());
        let mut names = Vec::with_capacity(self.config.features.len());

        for feature_def in &self.config.features {
            let matched = match &feature_def.predicate_selector {
                Some(selector) => predicate
                    .as_ref()
                    .map(|p| p.contains(selector))
                    .unwrap_or(false),
                None => true, // No selector means match all events
            };

            let value = match feature_def.aggregation {
                FeatureAggregation::Count => {
                    // Count matching events in the window (regardless of current event)
                    match &feature_def.predicate_selector {
                        Some(selector) => history
                            .iter()
                            .filter(|e| {
                                e.predicate
                                    .as_ref()
                                    .map(|p| p.contains(selector))
                                    .unwrap_or(false)
                            })
                            .count() as f64,
                        None => history_len as f64,
                    }
                }
                FeatureAggregation::Latest => {
                    if matched {
                        1.0
                    } else {
                        0.0
                    }
                }
                FeatureAggregation::Sum => {
                    let running = self.running_values.read();
                    running
                        .get(&feature_def.name)
                        .map(|v| v.iter().sum())
                        .unwrap_or(0.0)
                }
                FeatureAggregation::Mean => {
                    if history_len > 0 {
                        match &feature_def.predicate_selector {
                            Some(selector) => {
                                let count = history
                                    .iter()
                                    .filter(|e| {
                                        e.predicate
                                            .as_ref()
                                            .map(|p| p.contains(selector))
                                            .unwrap_or(false)
                                    })
                                    .count();
                                count as f64 / history_len as f64
                            }
                            None => 1.0, // All events match
                        }
                    } else {
                        0.0
                    }
                }
            };

            values.push(value);
            names.push(feature_def.name.clone());
        }

        // Update running values for matched event
        {
            let mut running = self.running_values.write();
            for feature_def in &self.config.features {
                let entry = running.entry(feature_def.name.clone()).or_default();
                let matched = match &feature_def.predicate_selector {
                    Some(selector) => predicate
                        .as_ref()
                        .map(|p| p.contains(selector))
                        .unwrap_or(false),
                    None => true,
                };
                entry.push_back(if matched { 1.0 } else { 0.0 });
                while entry.len() > self.config.window_size {
                    entry.pop_front();
                }
            }
        }

        ExtractedFeatures {
            values: Array1::from_vec(values),
            names,
            extracted_at: Utc::now(),
            event_id: event_id.to_string(),
        }
    }

    /// Resets the extractor state.
    pub fn reset(&self) {
        self.history.write().clear();
        self.running_values.write().clear();
    }

    /// Returns the current window size.
    pub fn current_window_size(&self) -> usize {
        self.history.read().len()
    }

    /// Returns the event type name.
    fn event_type_name(event: &StreamEvent) -> String {
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
            _ => "Other",
        }
        .to_string()
    }

    /// Extracts the predicate from a stream event, if it has one.
    fn extract_predicate(event: &StreamEvent) -> Option<String> {
        match event {
            StreamEvent::TripleAdded { predicate, .. }
            | StreamEvent::TripleRemoved { predicate, .. }
            | StreamEvent::QuadAdded { predicate, .. }
            | StreamEvent::QuadRemoved { predicate, .. } => Some(predicate.clone()),
            _ => None,
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventMetadata;
    use std::time::Duration;

    fn make_metadata(id: &str) -> EventMetadata {
        EventMetadata {
            event_id: id.to_string(),
            timestamp: Utc::now(),
            source: "test".to_string(),
            user: None,
            context: None,
            caused_by: None,
            version: "1.0".to_string(),
            properties: HashMap::new(),
            checksum: None,
        }
    }

    fn make_triple_event(id: &str, predicate: &str) -> StreamEvent {
        StreamEvent::TripleAdded {
            subject: "http://example.org/s".to_string(),
            predicate: predicate.to_string(),
            object: "http://example.org/o".to_string(),
            graph: None,
            metadata: make_metadata(id),
        }
    }

    // ── StreamingModelRunner Tests ───────────────────────────────────────────

    #[test]
    fn test_model_runner_basic_predict() {
        let config = ModelConfig {
            input_features: 3,
            batch_size: 10,
            max_latency_ms: 1000,
            ..Default::default()
        };
        let runner = StreamingModelRunner::new(config);

        let events = vec![
            (Array1::from_vec(vec![1.0, 2.0, 3.0]), "evt-1".to_string()),
            (Array1::from_vec(vec![4.0, 5.0, 6.0]), "evt-2".to_string()),
        ];
        let predictions = runner.predict(&events);
        assert_eq!(predictions.len(), 2);
        assert!(predictions[0].value.is_finite());
        assert!(predictions[0].confidence >= 0.0 && predictions[0].confidence <= 1.0);
    }

    #[test]
    fn test_model_runner_batch_trigger_by_size() {
        let config = ModelConfig {
            input_features: 2,
            batch_size: 3,
            max_latency_ms: 60_000,
            ..Default::default()
        };
        let runner = StreamingModelRunner::new(config);

        // Enqueue 2 events: no batch yet
        let result1 = runner.enqueue(Array1::from_vec(vec![1.0, 2.0]), "e1".to_string());
        assert!(result1.is_none());
        let result2 = runner.enqueue(Array1::from_vec(vec![3.0, 4.0]), "e2".to_string());
        assert!(result2.is_none());
        assert_eq!(runner.pending_count(), 2);

        // Third event triggers batch
        let result3 = runner.enqueue(Array1::from_vec(vec![5.0, 6.0]), "e3".to_string());
        assert!(result3.is_some());
        let predictions = result3.expect("should have predictions");
        assert_eq!(predictions.len(), 3);
        assert_eq!(runner.pending_count(), 0);
    }

    #[test]
    fn test_model_runner_flush() {
        let config = ModelConfig {
            input_features: 2,
            batch_size: 100,
            max_latency_ms: 60_000,
            ..Default::default()
        };
        let runner = StreamingModelRunner::new(config);

        runner.enqueue(Array1::from_vec(vec![1.0, 2.0]), "e1".to_string());
        runner.enqueue(Array1::from_vec(vec![3.0, 4.0]), "e2".to_string());

        let predictions = runner.flush();
        assert_eq!(predictions.len(), 2);
        assert_eq!(runner.pending_count(), 0);
    }

    #[test]
    fn test_model_runner_flush_if_due() {
        let config = ModelConfig {
            input_features: 2,
            batch_size: 100,
            max_latency_ms: 10, // 10ms
            ..Default::default()
        };
        let runner = StreamingModelRunner::new(config);

        runner.enqueue(Array1::from_vec(vec![1.0, 2.0]), "e1".to_string());
        std::thread::sleep(Duration::from_millis(20));

        let result = runner.flush_if_due();
        assert!(result.is_some());
    }

    #[test]
    fn test_model_runner_wrong_dimensions_ignored() {
        let config = ModelConfig {
            input_features: 3,
            ..Default::default()
        };
        let runner = StreamingModelRunner::new(config);
        let result = runner.enqueue(Array1::from_vec(vec![1.0, 2.0]), "bad".to_string());
        assert!(result.is_none());
        assert_eq!(runner.pending_count(), 0);
    }

    #[test]
    fn test_model_runner_update_weights() {
        let config = ModelConfig {
            input_features: 2,
            ..Default::default()
        };
        let runner = StreamingModelRunner::new(config);
        runner.update_weights(Array1::from_vec(vec![1.0, 2.0]), 0.5);

        let predictions = runner.predict(&[(Array1::from_vec(vec![1.0, 1.0]), "e1".to_string())]);
        // value = 0.5 + 1.0*1.0 + 2.0*1.0 = 3.5
        assert!((predictions[0].value - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_model_runner_stats() {
        let config = ModelConfig {
            input_features: 2,
            batch_size: 2,
            ..Default::default()
        };
        let runner = StreamingModelRunner::new(config);
        runner.enqueue(Array1::from_vec(vec![1.0, 2.0]), "e1".to_string());
        runner.enqueue(Array1::from_vec(vec![3.0, 4.0]), "e2".to_string());

        let stats = runner.stats();
        assert_eq!(stats.events_processed, 2);
        assert_eq!(stats.batches_executed, 1);
        assert_eq!(stats.size_triggered_batches, 1);
    }

    // ── StreamAnomalyDetector Tests ──────────────────────────────────────────

    #[test]
    fn test_anomaly_detector_normal_values() {
        let config = AnomalyDetectorConfig {
            sigma_threshold: 3.0,
            window_size: 50,
            min_samples: 5,
            adaptive_rate: 0.0,
        };
        let detector = StreamAnomalyDetector::new(config);

        // Feed normal values
        for i in 0..20 {
            let result = detector.is_anomaly(100.0 + (i as f64 * 0.1));
            if i >= 5 {
                assert!(
                    !result.is_anomaly,
                    "normal value should not be anomaly at i={}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_anomaly_detector_detects_outlier() {
        let config = AnomalyDetectorConfig {
            sigma_threshold: 3.0,
            window_size: 100,
            min_samples: 10,
            adaptive_rate: 0.0,
        };
        let detector = StreamAnomalyDetector::new(config);

        // Feed stable values
        for _ in 0..30 {
            detector.is_anomaly(100.0);
        }

        // Feed a huge outlier
        let result = detector.is_anomaly(10000.0);
        assert!(result.is_anomaly);
        assert!(result.z_score > 3.0);
    }

    #[test]
    fn test_anomaly_detector_insufficient_samples() {
        let config = AnomalyDetectorConfig {
            min_samples: 20,
            ..Default::default()
        };
        let detector = StreamAnomalyDetector::new(config);

        // Not enough samples yet
        let result = detector.is_anomaly(999999.0);
        assert!(!result.is_anomaly);
        assert_eq!(result.window_samples, 1);
    }

    #[test]
    fn test_anomaly_detector_sliding_window() {
        let config = AnomalyDetectorConfig {
            window_size: 10,
            min_samples: 5,
            sigma_threshold: 3.0,
            adaptive_rate: 0.0,
        };
        let detector = StreamAnomalyDetector::new(config);

        // Fill window with values around 100
        for _ in 0..10 {
            detector.is_anomaly(100.0);
        }

        // Now shift to values around 200 to fill the window
        for _ in 0..10 {
            detector.is_anomaly(200.0);
        }

        // After window shift, 200 should be normal
        let result = detector.is_anomaly(200.0);
        assert!(!result.is_anomaly);
        assert!((result.window_mean - 200.0).abs() < 1.0);
    }

    #[test]
    fn test_anomaly_detector_adaptive_threshold() {
        let config = AnomalyDetectorConfig {
            sigma_threshold: 3.0,
            adaptive_rate: 1.0,
            ..Default::default()
        };
        let detector = StreamAnomalyDetector::new(config);

        let initial_threshold = detector.effective_threshold();
        detector.feedback(false); // false positive -> raise threshold
        let new_threshold = detector.effective_threshold();
        assert!(new_threshold > initial_threshold);

        detector.feedback(true); // true positive -> lower threshold
        let final_threshold = detector.effective_threshold();
        assert!(final_threshold < new_threshold);
    }

    #[test]
    fn test_anomaly_detector_stats() {
        let config = AnomalyDetectorConfig {
            sigma_threshold: 2.0,
            min_samples: 3,
            window_size: 20,
            adaptive_rate: 0.0,
        };
        let detector = StreamAnomalyDetector::new(config);

        for _ in 0..10 {
            detector.is_anomaly(50.0);
        }
        detector.is_anomaly(9999.0); // anomaly

        let stats = detector.stats();
        assert_eq!(stats.values_processed, 11);
        assert!(stats.anomalies_detected >= 1);
    }

    #[test]
    fn test_anomaly_detector_reset() {
        let detector = StreamAnomalyDetector::new(AnomalyDetectorConfig::default());
        for _ in 0..20 {
            detector.is_anomaly(100.0);
        }
        detector.reset();
        let stats = detector.stats();
        assert_eq!(stats.values_processed, 0);
    }

    // ── StreamFeatureExtractor Tests ─────────────────────────────────────────

    #[test]
    fn test_feature_extractor_basic() {
        let config = FeatureExtractorConfig::default();
        let extractor = StreamFeatureExtractor::new(config);

        let event = make_triple_event("e1", "http://example.org/name");
        let features = extractor.extract(&event, "e1");
        assert!(!features.values.is_empty());
        assert_eq!(features.values.len(), features.names.len());
    }

    #[test]
    fn test_feature_extractor_predicate_selector() {
        let config = FeatureExtractorConfig {
            features: vec![
                FeatureDefinition {
                    name: "name_events".to_string(),
                    predicate_selector: Some("name".to_string()),
                    aggregation: FeatureAggregation::Count,
                },
                FeatureDefinition {
                    name: "age_events".to_string(),
                    predicate_selector: Some("age".to_string()),
                    aggregation: FeatureAggregation::Count,
                },
            ],
            window_size: 100,
        };
        let extractor = StreamFeatureExtractor::new(config);

        // Add events with "name" predicate
        for i in 0..3 {
            let event = make_triple_event(&format!("n{}", i), "http://example.org/name");
            extractor.extract(&event, &format!("n{}", i));
        }

        // Add events with "age" predicate
        let event = make_triple_event("a1", "http://example.org/age");
        let features = extractor.extract(&event, "a1");

        // name_events should be 3, age_events should be 1
        assert_eq!(features.names[0], "name_events");
        assert!((features.values[0] - 3.0).abs() < 0.01);
        assert_eq!(features.names[1], "age_events");
        assert!((features.values[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_feature_extractor_mean_aggregation() {
        let config = FeatureExtractorConfig {
            features: vec![FeatureDefinition {
                name: "ratio".to_string(),
                predicate_selector: Some("type".to_string()),
                aggregation: FeatureAggregation::Mean,
            }],
            window_size: 10,
        };
        let extractor = StreamFeatureExtractor::new(config);

        // 2 matching out of 4 total
        extractor.extract(&make_triple_event("e1", "http://ex/type"), "e1");
        extractor.extract(&make_triple_event("e2", "http://ex/name"), "e2");
        extractor.extract(&make_triple_event("e3", "http://ex/type"), "e3");
        let features = extractor.extract(&make_triple_event("e4", "http://ex/name"), "e4");

        // 2 matching out of 4 = 0.5 ratio
        assert!((features.values[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_feature_extractor_window_eviction() {
        let config = FeatureExtractorConfig {
            features: vec![FeatureDefinition {
                name: "count".to_string(),
                predicate_selector: None,
                aggregation: FeatureAggregation::Count,
            }],
            window_size: 3,
        };
        let extractor = StreamFeatureExtractor::new(config);

        for i in 0..5 {
            extractor.extract(
                &make_triple_event(&format!("e{}", i), "http://ex/p"),
                &format!("e{}", i),
            );
        }

        assert_eq!(extractor.current_window_size(), 3);
    }

    #[test]
    fn test_feature_extractor_reset() {
        let extractor = StreamFeatureExtractor::new(FeatureExtractorConfig::default());
        extractor.extract(&make_triple_event("e1", "http://ex/p"), "e1");
        extractor.reset();
        assert_eq!(extractor.current_window_size(), 0);
    }

    #[test]
    fn test_feature_extractor_non_triple_events() {
        let config = FeatureExtractorConfig::default();
        let extractor = StreamFeatureExtractor::new(config);

        let event = StreamEvent::SchemaChanged {
            schema_type: crate::event::SchemaType::Ontology,
            change_type: crate::event::SchemaChangeType::Added,
            details: "test".to_string(),
            metadata: make_metadata("schema-1"),
        };
        let features = extractor.extract(&event, "schema-1");
        assert!(!features.values.is_empty());
    }
}
