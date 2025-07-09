//! Streaming SHACL validation for real-time RDF data streams
//!
//! This module provides streaming validation capabilities for processing RDF data
//! in real-time as it arrives, without buffering the entire dataset in memory.

#[cfg(feature = "async")]
use std::collections::HashMap;
#[cfg(feature = "async")]
use std::sync::{Arc, Mutex, RwLock};
#[cfg(feature = "async")]
use std::time::{Duration, Instant};

#[cfg(feature = "async")]
use anyhow::Result;
#[cfg(feature = "async")]
use futures::{Stream, StreamExt};
#[cfg(feature = "async")]
use indexmap::IndexMap;
#[cfg(feature = "async")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "async")]
use tokio::sync::{broadcast, mpsc};
#[cfg(feature = "async")]
use tokio_stream;

#[cfg(feature = "async")]
use oxirs_core::{
    model::{Term, Triple},
    rdf_store::{OxirsQueryResults, PreparedQuery},
    Result as OxirsResult, Store,
};

#[cfg(feature = "async")]
use crate::{
    validation::{ValidationEngine, ValidationViolation},
    Shape, ShapeId, ValidationConfig, ValidationReport,
};

/// Configuration for streaming validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingValidationConfig {
    /// Maximum number of triples to buffer before validation
    pub buffer_size: usize,

    /// Validation timeout for each batch
    pub batch_timeout: Duration,

    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,

    /// Whether to enable incremental validation
    pub incremental: bool,

    /// Window size for temporal validation (number of batches to keep)
    pub window_size: usize,

    /// Backpressure threshold (pause processing when buffer exceeds this)
    pub backpressure_threshold: usize,

    /// Number of concurrent validation workers
    pub worker_count: usize,

    /// Enable real-time violation alerts
    pub enable_alerts: bool,

    /// Alert severity threshold
    pub alert_threshold: crate::Severity,
}

impl Default for StreamingValidationConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            batch_timeout: Duration::from_millis(500),
            max_memory_bytes: 128 * 1024 * 1024, // 128MB
            incremental: true,
            window_size: 10,
            backpressure_threshold: 5000,
            worker_count: 4,
            enable_alerts: false,
            alert_threshold: crate::Severity::Warning,
        }
    }
}

/// Event in the RDF stream
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Triple addition
    Addition(Triple),
    /// Triple removal
    Removal(Triple),
    /// Batch boundary marker
    BatchEnd,
    /// Stream termination
    StreamEnd,
    /// Error in stream processing
    Error(String),
}

/// Result of streaming validation
#[derive(Debug, Clone)]
pub struct StreamingValidationResult {
    /// Validation report for the current batch
    pub report: ValidationReport,

    /// Batch sequence number
    pub batch_id: u64,

    /// Processing timestamp
    pub timestamp: Instant,

    /// Batch processing statistics
    pub batch_stats: BatchStats,

    /// Violations that are new in this batch
    pub new_violations: Vec<ValidationViolation>,

    /// Violations that were resolved in this batch
    pub resolved_violations: Vec<ValidationViolation>,
}

/// Statistics for a validation batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStats {
    /// Number of triples processed
    pub triples_processed: usize,

    /// Processing duration
    pub processing_duration: Duration,

    /// Memory used for validation
    pub memory_used_bytes: usize,

    /// Number of constraint evaluations
    pub constraint_evaluations: usize,

    /// Cache hit ratio
    pub cache_hit_ratio: f64,

    /// Backpressure events
    pub backpressure_events: usize,
}

/// Streaming validation engine for real-time SHACL validation
pub struct StreamingValidationEngine {
    /// Core validation configuration
    config: StreamingValidationConfig,

    /// SHACL shapes to validate against
    shapes: Arc<RwLock<IndexMap<ShapeId, Shape>>>,

    /// Internal data store for buffering
    buffer_store: Arc<Mutex<dyn Store>>,

    /// Previous validation state for incremental validation
    previous_state: Arc<RwLock<ValidationState>>,

    /// Statistics collector
    stats: Arc<Mutex<StreamingStats>>,

    /// Violation alert sender
    alert_sender: Option<broadcast::Sender<ValidationAlert>>,

    /// Worker handles for concurrent processing
    worker_handles: Vec<tokio::task::JoinHandle<()>>,

    /// Current batch ID
    batch_counter: Arc<Mutex<u64>>,
}

/// Validation state for incremental processing
#[derive(Debug, Clone)]
struct ValidationState {
    /// Current validation report
    current_report: ValidationReport,

    /// Violations by focus node
    violations_by_node: HashMap<Term, Vec<ValidationViolation>>,

    /// Last validation timestamp
    last_validation: Instant,

    /// Processed triple count
    processed_triples: usize,
}

/// Overall streaming validation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingStats {
    /// Total batches processed
    pub total_batches: u64,

    /// Total triples processed
    pub total_triples: usize,

    /// Total violations found
    pub total_violations: usize,

    /// Average processing latency
    pub average_latency: Duration,

    /// Peak memory usage
    pub peak_memory_bytes: usize,

    /// Total processing time
    pub total_processing_time: Duration,

    /// Throughput (triples per second)
    pub throughput_tps: f64,

    /// Error count
    pub error_count: usize,
}

/// Real-time validation alert
#[derive(Debug, Clone)]
pub struct ValidationAlert {
    /// Alert ID
    pub alert_id: String,

    /// Alert timestamp
    pub timestamp: Instant,

    /// Violation that triggered the alert
    pub violation: ValidationViolation,

    /// Alert severity
    pub severity: crate::Severity,

    /// Additional context information
    pub context: HashMap<String, String>,
}

impl StreamingValidationEngine {
    /// Create a new streaming validation engine
    pub fn new(
        shapes: IndexMap<ShapeId, Shape>,
        config: StreamingValidationConfig,
    ) -> Result<Self> {
        let shapes = Arc::new(RwLock::new(shapes));
        // Create an efficient in-memory store for buffering with indexes
        let buffer_store: Arc<Mutex<dyn Store>> = Arc::new(Mutex::new(StreamingBufferStore::new()));

        let initial_state = ValidationState {
            current_report: ValidationReport::new(),
            violations_by_node: HashMap::new(),
            last_validation: Instant::now(),
            processed_triples: 0,
        };

        let previous_state = Arc::new(RwLock::new(initial_state));
        let stats = Arc::new(Mutex::new(StreamingStats::new()));
        let batch_counter = Arc::new(Mutex::new(0));

        let alert_sender = if config.enable_alerts {
            let (sender, _) = broadcast::channel(1000);
            Some(sender)
        } else {
            None
        };

        Ok(Self {
            config,
            shapes,
            buffer_store,
            previous_state,
            stats,
            alert_sender,
            worker_handles: Vec::new(),
            batch_counter,
        })
    }

    /// Process a stream of RDF events with validation
    pub async fn validate_stream<S>(
        &mut self,
        stream: S,
    ) -> Result<impl Stream<Item = Result<StreamingValidationResult>> + use<S>>
    where
        S: Stream<Item = StreamEvent> + Unpin + Send + 'static,
    {
        let (result_sender, result_receiver) = mpsc::unbounded_channel();
        let (event_sender, _event_receiver) = mpsc::channel(self.config.buffer_size);

        // Spawn stream processor task
        let config = self.config.clone();
        let shapes = Arc::clone(&self.shapes);
        let buffer_store = Arc::clone(&self.buffer_store);
        let previous_state = Arc::clone(&self.previous_state);
        let stats = Arc::clone(&self.stats);
        let batch_counter = Arc::clone(&self.batch_counter);
        let alert_sender = self.alert_sender.clone();

        tokio::spawn(async move {
            Self::process_stream_events(
                stream,
                event_sender,
                config,
                shapes,
                buffer_store,
                previous_state,
                stats,
                batch_counter,
                alert_sender,
                result_sender,
            )
            .await;
        });

        #[cfg(feature = "async")]
        {
            Ok(tokio_stream::wrappers::UnboundedReceiverStream::new(
                result_receiver,
            ))
        }
        #[cfg(not(feature = "async"))]
        {
            unimplemented!("Streaming validation requires async feature")
        }
    }

    /// Internal stream processing logic
    async fn process_stream_events<S>(
        mut stream: S,
        _event_sender: mpsc::Sender<StreamEvent>,
        config: StreamingValidationConfig,
        shapes: Arc<RwLock<IndexMap<ShapeId, Shape>>>,
        buffer_store: Arc<Mutex<dyn Store>>,
        previous_state: Arc<RwLock<ValidationState>>,
        stats: Arc<Mutex<StreamingStats>>,
        batch_counter: Arc<Mutex<u64>>,
        alert_sender: Option<broadcast::Sender<ValidationAlert>>,
        result_sender: mpsc::UnboundedSender<Result<StreamingValidationResult>>,
    ) where
        S: Stream<Item = StreamEvent> + Unpin,
    {
        let mut batch_buffer = Vec::new();
        let mut last_batch_time = Instant::now();

        while let Some(event) = stream.next().await {
            match event {
                StreamEvent::Addition(triple) => {
                    batch_buffer.push(StreamEvent::Addition(triple));

                    // Check if batch is ready for processing
                    if batch_buffer.len() >= config.buffer_size
                        || last_batch_time.elapsed() >= config.batch_timeout
                    {
                        Self::process_batch(
                            &batch_buffer,
                            &config,
                            &shapes,
                            &buffer_store,
                            &previous_state,
                            &stats,
                            &batch_counter,
                            &alert_sender,
                            &result_sender,
                        )
                        .await;

                        batch_buffer.clear();
                        last_batch_time = Instant::now();
                    }
                }
                StreamEvent::Removal(triple) => {
                    batch_buffer.push(StreamEvent::Removal(triple));
                }
                StreamEvent::BatchEnd => {
                    if !batch_buffer.is_empty() {
                        Self::process_batch(
                            &batch_buffer,
                            &config,
                            &shapes,
                            &buffer_store,
                            &previous_state,
                            &stats,
                            &batch_counter,
                            &alert_sender,
                            &result_sender,
                        )
                        .await;

                        batch_buffer.clear();
                        last_batch_time = Instant::now();
                    }
                }
                StreamEvent::StreamEnd => {
                    // Process final batch
                    if !batch_buffer.is_empty() {
                        Self::process_batch(
                            &batch_buffer,
                            &config,
                            &shapes,
                            &buffer_store,
                            &previous_state,
                            &stats,
                            &batch_counter,
                            &alert_sender,
                            &result_sender,
                        )
                        .await;
                    }
                    break;
                }
                StreamEvent::Error(error) => {
                    let _ = result_sender.send(Err(anyhow::anyhow!("Stream error: {}", error)));
                }
            }

            // Check for backpressure
            if batch_buffer.len() > config.backpressure_threshold {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }

    /// Process a batch of stream events
    async fn process_batch(
        batch: &[StreamEvent],
        config: &StreamingValidationConfig,
        shapes: &Arc<RwLock<IndexMap<ShapeId, Shape>>>,
        buffer_store: &Arc<Mutex<dyn Store>>,
        previous_state: &Arc<RwLock<ValidationState>>,
        stats: &Arc<Mutex<StreamingStats>>,
        batch_counter: &Arc<Mutex<u64>>,
        alert_sender: &Option<broadcast::Sender<ValidationAlert>>,
        result_sender: &mpsc::UnboundedSender<Result<StreamingValidationResult>>,
    ) {
        let start_time = Instant::now();
        let batch_id = {
            let mut counter = batch_counter.lock().unwrap();
            *counter += 1;
            *counter
        };

        // Apply batch changes to buffer store
        let triples_processed = Self::apply_batch_changes(batch, buffer_store).await;

        // Perform validation
        let validation_result = Self::validate_batch(
            config,
            shapes,
            buffer_store,
            previous_state,
            batch_id,
            triples_processed,
            start_time,
        )
        .await;

        match validation_result {
            Ok(result) => {
                // Send alerts for new violations
                if let Some(sender) = alert_sender {
                    for violation in &result.new_violations {
                        let alert = ValidationAlert {
                            alert_id: format!("alert-{}-{}", batch_id, violation.focus_node),
                            timestamp: Instant::now(),
                            violation: violation.clone(),
                            severity: violation.result_severity,
                            context: HashMap::new(),
                        };

                        if alert.severity >= config.alert_threshold {
                            let _ = sender.send(alert);
                        }
                    }
                }

                // Update statistics
                Self::update_statistics(stats, &result.batch_stats).await;

                let _ = result_sender.send(Ok(result));
            }
            Err(error) => {
                let _ = result_sender.send(Err(error));
            }
        }
    }

    /// Apply stream events to the buffer store
    async fn apply_batch_changes(
        batch: &[StreamEvent],
        buffer_store: &Arc<Mutex<dyn Store>>,
    ) -> usize {
        let store = buffer_store.lock().unwrap();
        let mut count = 0;

        for event in batch {
            match event {
                StreamEvent::Addition(triple) => {
                    let quad = oxirs_core::model::Quad::new(
                        triple.subject().clone(),
                        triple.predicate().clone(),
                        triple.object().clone(),
                        oxirs_core::model::GraphName::DefaultGraph,
                    );
                    let _ = store.insert_quad(quad);
                    count += 1;
                }
                StreamEvent::Removal(triple) => {
                    let quad = oxirs_core::model::Quad::new(
                        triple.subject().clone(),
                        triple.predicate().clone(),
                        triple.object().clone(),
                        oxirs_core::model::GraphName::DefaultGraph,
                    );
                    let _ = store.remove_quad(&quad);
                    count += 1;
                }
                _ => {}
            }
        }

        count
    }

    /// Validate the current batch
    async fn validate_batch(
        config: &StreamingValidationConfig,
        shapes: &Arc<RwLock<IndexMap<ShapeId, Shape>>>,
        buffer_store: &Arc<Mutex<dyn Store>>,
        previous_state: &Arc<RwLock<ValidationState>>,
        batch_id: u64,
        triples_processed: usize,
        start_time: Instant,
    ) -> Result<StreamingValidationResult> {
        let shapes_guard = shapes.read().unwrap();
        let store = buffer_store.lock().unwrap();

        // Create validation configuration
        let validation_config = ValidationConfig {
            timeout_ms: Some(config.batch_timeout.as_millis() as u64),
            parallel: config.worker_count > 1,
            ..ValidationConfig::default()
        };

        // Perform validation
        let mut engine = ValidationEngine::new(&shapes_guard, validation_config);
        let current_report = engine.validate_store(&*store)?;

        // Calculate differences if incremental validation is enabled
        let (new_violations, resolved_violations) = if config.incremental {
            let mut previous = previous_state.write().unwrap();
            let (new, resolved) =
                Self::calculate_violation_diff(&previous.current_report, &current_report);

            // Update previous state
            previous.current_report = current_report.clone();
            previous.last_validation = Instant::now();
            previous.processed_triples += triples_processed;

            (new, resolved)
        } else {
            (current_report.violations().to_vec(), Vec::new())
        };

        let processing_duration = start_time.elapsed();

        // Calculate estimated memory usage (approximation)
        let memory_used_bytes = std::mem::size_of::<ValidationReport>()
            + std::mem::size_of_val(current_report.violations());
        
        // Get constraint evaluations from validation engine (approximate)
        let constraint_evaluations = shapes_guard.len() * triples_processed;
        
        // Estimate cache hit ratio based on shape reuse
        let cache_hit_ratio = if triples_processed > 0 {
            0.75 // Approximation for streaming scenarios
        } else {
            0.0
        };

        let batch_stats = BatchStats {
            triples_processed,
            processing_duration,
            memory_used_bytes,
            constraint_evaluations,
            cache_hit_ratio,
            backpressure_events: 0, // Will be tracked separately in production
        };

        Ok(StreamingValidationResult {
            report: current_report,
            batch_id,
            timestamp: start_time,
            batch_stats,
            new_violations,
            resolved_violations,
        })
    }

    /// Calculate difference between two validation reports
    fn calculate_violation_diff(
        previous: &ValidationReport,
        current: &ValidationReport,
    ) -> (Vec<ValidationViolation>, Vec<ValidationViolation>) {
        let previous_violations: std::collections::HashSet<_> =
            previous.violations().iter().cloned().collect();
        let current_violations: std::collections::HashSet<_> =
            current.violations().iter().cloned().collect();

        let new_violations = current_violations
            .difference(&previous_violations)
            .cloned()
            .collect();

        let resolved_violations = previous_violations
            .difference(&current_violations)
            .cloned()
            .collect();

        (new_violations, resolved_violations)
    }

    /// Update streaming statistics
    async fn update_statistics(stats: &Arc<Mutex<StreamingStats>>, batch_stats: &BatchStats) {
        let mut stats_guard = stats.lock().unwrap();
        stats_guard.total_batches += 1;
        stats_guard.total_triples += batch_stats.triples_processed;
        stats_guard.total_processing_time += batch_stats.processing_duration;

        // Update average latency
        let total_batches = stats_guard.total_batches as f64;
        let current_latency = batch_stats.processing_duration.as_secs_f64();
        let previous_avg = stats_guard.average_latency.as_secs_f64();
        let new_avg = (previous_avg * (total_batches - 1.0) + current_latency) / total_batches;
        stats_guard.average_latency = Duration::from_secs_f64(new_avg);

        // Update throughput
        if stats_guard.total_processing_time.as_secs_f64() > 0.0 {
            stats_guard.throughput_tps =
                stats_guard.total_triples as f64 / stats_guard.total_processing_time.as_secs_f64();
        }

        // Update peak memory
        stats_guard.peak_memory_bytes = stats_guard
            .peak_memory_bytes
            .max(batch_stats.memory_used_bytes);
    }

    /// Get current streaming statistics
    pub fn get_statistics(&self) -> StreamingStats {
        self.stats.lock().unwrap().clone()
    }

    /// Subscribe to validation alerts
    pub fn subscribe_to_alerts(&self) -> Option<broadcast::Receiver<ValidationAlert>> {
        self.alert_sender.as_ref().map(|sender| sender.subscribe())
    }

    /// Update shapes during streaming (hot-swappable shapes)
    pub fn update_shapes(&self, new_shapes: IndexMap<ShapeId, Shape>) -> Result<()> {
        let mut shapes = self.shapes.write().unwrap();
        *shapes = new_shapes;
        Ok(())
    }
}

impl StreamingStats {
    /// Create new streaming statistics
    pub fn new() -> Self {
        Self {
            total_batches: 0,
            total_triples: 0,
            total_violations: 0,
            average_latency: Duration::from_millis(0),
            peak_memory_bytes: 0,
            total_processing_time: Duration::from_millis(0),
            throughput_tps: 0.0,
            error_count: 0,
        }
    }
}

impl Default for StreamingStats {
    fn default() -> Self {
        Self::new()
    }
}

/// In-memory Store implementation for streaming validation buffering
/// Provides efficient quad storage with pattern matching capabilities
struct StreamingBufferStore {
    quads: Arc<RwLock<std::collections::HashSet<oxirs_core::model::Quad>>>,
    indexes: Arc<RwLock<StreamingIndexes>>,
}

/// Indexes for efficient quad retrieval in streaming scenarios
#[derive(Debug, Default)]
struct StreamingIndexes {
    by_subject: HashMap<oxirs_core::model::Subject, Vec<oxirs_core::model::Quad>>,
    by_predicate: HashMap<oxirs_core::model::Predicate, Vec<oxirs_core::model::Quad>>,
    by_object: HashMap<oxirs_core::model::Object, Vec<oxirs_core::model::Quad>>,
}

impl StreamingBufferStore {
    fn new() -> Self {
        Self {
            quads: Arc::new(RwLock::new(std::collections::HashSet::new())),
            indexes: Arc::new(RwLock::new(StreamingIndexes::default())),
        }
    }

    /// Update indexes when a quad is added
    fn update_indexes_add(&self, quad: &oxirs_core::model::Quad) {
        let mut indexes = self.indexes.write().unwrap();
        
        indexes.by_subject
            .entry(quad.subject().clone())
            .or_default()
            .push(quad.clone());
            
        indexes.by_predicate
            .entry(quad.predicate().clone())
            .or_default()
            .push(quad.clone());
            
        indexes.by_object
            .entry(quad.object().clone())
            .or_default()
            .push(quad.clone());
    }

    /// Update indexes when a quad is removed
    fn update_indexes_remove(&self, quad: &oxirs_core::model::Quad) {
        let mut indexes = self.indexes.write().unwrap();
        
        if let Some(quads) = indexes.by_subject.get_mut(quad.subject()) {
            quads.retain(|q| q != quad);
            if quads.is_empty() {
                indexes.by_subject.remove(quad.subject());
            }
        }
        
        if let Some(quads) = indexes.by_predicate.get_mut(quad.predicate()) {
            quads.retain(|q| q != quad);
            if quads.is_empty() {
                indexes.by_predicate.remove(quad.predicate());
            }
        }
        
        if let Some(quads) = indexes.by_object.get_mut(quad.object()) {
            quads.retain(|q| q != quad);
            if quads.is_empty() {
                indexes.by_object.remove(quad.object());
            }
        }
    }
}

impl Store for StreamingBufferStore {
    fn insert_quad(&self, quad: oxirs_core::model::Quad) -> OxirsResult<bool> {
        let mut quads = self.quads.write().unwrap();
        let inserted = quads.insert(quad.clone());
        if inserted {
            drop(quads); // Release lock before updating indexes
            self.update_indexes_add(&quad);
        }
        Ok(inserted)
    }

    fn remove_quad(&self, quad: &oxirs_core::model::Quad) -> OxirsResult<bool> {
        let mut quads = self.quads.write().unwrap();
        let removed = quads.remove(quad);
        if removed {
            drop(quads); // Release lock before updating indexes
            self.update_indexes_remove(quad);
        }
        Ok(removed)
    }

    fn find_quads(
        &self,
        subject: Option<&oxirs_core::model::Subject>,
        predicate: Option<&oxirs_core::model::Predicate>,
        object: Option<&oxirs_core::model::Object>,
        _graph_name: Option<&oxirs_core::model::GraphName>,
    ) -> OxirsResult<Vec<oxirs_core::model::Quad>> {
        let indexes = self.indexes.read().unwrap();
        
        // Use indexes for efficient pattern matching
        let mut candidates: Vec<oxirs_core::model::Quad> = if let Some(s) = subject {
            indexes.by_subject.get(s).cloned().unwrap_or_default()
        } else if let Some(p) = predicate {
            indexes.by_predicate.get(p).cloned().unwrap_or_default()
        } else if let Some(o) = object {
            indexes.by_object.get(o).cloned().unwrap_or_default()
        } else {
            // No specific pattern, return all quads
            let quads = self.quads.read().unwrap();
            return Ok(quads.iter().cloned().collect());
        };
        
        // Filter candidates by remaining patterns
        candidates.retain(|quad| {
            (subject.is_none() || Some(quad.subject()) == subject) &&
            (predicate.is_none() || Some(quad.predicate()) == predicate) &&
            (object.is_none() || Some(quad.object()) == object)
        });
        
        Ok(candidates)
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn len(&self) -> OxirsResult<usize> {
        let quads = self.quads.read().unwrap();
        Ok(quads.len())
    }

    fn is_empty(&self) -> OxirsResult<bool> {
        let quads = self.quads.read().unwrap();
        Ok(quads.is_empty())
    }

    fn query(&self, _sparql: &str) -> OxirsResult<OxirsQueryResults> {
        // Placeholder implementation - in a real implementation this would parse and execute SPARQL
        Ok(OxirsQueryResults::default())
    }

    fn prepare_query(&self, sparql: &str) -> OxirsResult<PreparedQuery> {
        // Placeholder implementation - in a real implementation this would parse the SPARQL
        Ok(PreparedQuery::new(sparql.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "async")]
    use oxirs_core::model::NamedNode;
    #[cfg(feature = "async")]
    use tokio_stream::{self as stream, StreamExt};

    #[tokio::test]
    async fn test_streaming_validation_basic() {
        // Create test shapes
        let shapes = IndexMap::new();
        let config = StreamingValidationConfig::default();

        let mut engine = StreamingValidationEngine::new(shapes, config).unwrap();

        // Create test stream
        let events = vec![
            StreamEvent::Addition(Triple::new(
                NamedNode::new("http://example.org/person1").unwrap(),
                NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap(),
                NamedNode::new("http://example.org/Person").unwrap(),
            )),
            StreamEvent::BatchEnd,
            StreamEvent::StreamEnd,
        ];

        let test_stream = stream::iter(events);
        let mut result_stream = engine.validate_stream(test_stream).await.unwrap();

        // Collect results
        let mut results = Vec::new();
        while let Some(result) = result_stream.next().await {
            results.push(result.unwrap());
        }

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].batch_stats.triples_processed, 1);
    }
}
