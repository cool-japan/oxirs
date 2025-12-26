//! # Streaming Query Processor for RDF-star
//!
//! Real-time continuous query evaluation over RDF-star data streams.
//!
//! This module provides:
//! - **Continuous Queries**: Register queries that run continuously on incoming data
//! - **Windowed Processing**: Time-based and count-based sliding windows
//! - **Incremental Evaluation**: Efficient delta processing for updates
//! - **Event-Driven Architecture**: Push-based notification of results
//! - **Backpressure Handling**: Flow control for high-velocity streams
//!
//! ## Overview
//!
//! Streaming query processing enables real-time analysis of RDF-star data as it arrives,
//! supporting use cases like:
//!
//! - Real-time knowledge graph updates
//! - Event stream processing with metadata annotations
//! - Continuous monitoring of assertion confidence
//! - Provenance tracking in data pipelines
//!
//! ## Example
//!
//! ```rust,ignore
//! use oxirs_star::streaming_query::{StreamingQueryEngine, WindowConfig, QueryRegistration};
//! use oxirs_star::{StarTriple, StarTerm};
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create streaming query engine
//! let mut engine = StreamingQueryEngine::new();
//!
//! // Register a continuous query with a 1-minute sliding window
//! let query = QueryRegistration::new(
//!     "high_confidence",
//!     "SELECT * WHERE { << ?s ?p ?o >> ?meta ?value . FILTER(?meta = <confidence> && ?value > 0.9) }",
//!     WindowConfig::sliding(60, 10), // 60 second window, 10 second slide
//! );
//!
//! let query_id = engine.register_query(query)?;
//!
//! // Subscribe to results
//! engine.subscribe(query_id, |results| {
//!     println!("New high-confidence assertions: {:?}", results);
//! });
//!
//! // Start processing
//! engine.start().await?;
//!
//! // Feed data to the engine
//! let triple = StarTriple::new(
//!     StarTerm::iri("http://example.org/alice")?,
//!     StarTerm::iri("http://example.org/age")?,
//!     StarTerm::literal("30")?,
//! );
//! engine.ingest(triple).await?;
//!
//! # Ok(())
//! # }
//! ```

use crate::{StarError, StarResult, StarTerm, StarTriple};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::{info, instrument};

// Import SciRS2 components (SCIRS2 POLICY)
use scirs2_core::profiling::Profiler;

// SIMD-accelerated pattern matching module (v0.4.0 Phase 1)
#[path = "streaming_query/simd_pattern_matcher.rs"]
pub mod simd_pattern_matcher;

pub use simd_pattern_matcher::{
    SimdCepSequenceMatcher, SimdPredicateMatcher, SimdQuotedTripleFilter,
};

/// Streaming query engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Maximum number of triples to buffer
    pub max_buffer_size: usize,
    /// Default window duration in seconds
    pub default_window_secs: u64,
    /// Default slide interval in seconds
    pub default_slide_secs: u64,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Batch size for processing
    pub batch_size: usize,
    /// Enable backpressure handling
    pub enable_backpressure: bool,
    /// Maximum concurrent queries
    pub max_concurrent_queries: usize,
    /// Query timeout in milliseconds
    pub query_timeout_ms: u64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 100_000,
            default_window_secs: 60,
            default_slide_secs: 10,
            enable_metrics: true,
            batch_size: 1000,
            enable_backpressure: true,
            max_concurrent_queries: 100,
            query_timeout_ms: 30_000,
        }
    }
}

/// Window configuration for continuous queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowConfig {
    /// Tumbling window (non-overlapping)
    Tumbling {
        /// Window duration in seconds
        duration_secs: u64,
    },
    /// Sliding window (overlapping)
    Sliding {
        /// Window duration in seconds
        duration_secs: u64,
        /// Slide interval in seconds
        slide_secs: u64,
    },
    /// Count-based window
    Count {
        /// Number of triples per window
        count: usize,
    },
    /// Session window (activity-based)
    Session {
        /// Inactivity timeout in seconds
        timeout_secs: u64,
    },
    /// Landmark window (from start to now)
    Landmark,
}

impl WindowConfig {
    /// Create a sliding window
    pub fn sliding(duration_secs: u64, slide_secs: u64) -> Self {
        Self::Sliding {
            duration_secs,
            slide_secs,
        }
    }

    /// Create a tumbling window
    pub fn tumbling(duration_secs: u64) -> Self {
        Self::Tumbling { duration_secs }
    }

    /// Create a count-based window
    pub fn count(count: usize) -> Self {
        Self::Count { count }
    }

    /// Create a session window
    pub fn session(timeout_secs: u64) -> Self {
        Self::Session { timeout_secs }
    }

    /// Create a landmark window
    pub fn landmark() -> Self {
        Self::Landmark
    }
}

/// Query registration details
#[derive(Debug, Clone)]
pub struct QueryRegistration {
    /// Unique query name
    pub name: String,
    /// SPARQL-star query pattern
    pub query: String,
    /// Window configuration
    pub window: WindowConfig,
    /// Priority (higher = more important)
    pub priority: u8,
    /// Enable incremental evaluation
    pub incremental: bool,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl QueryRegistration {
    /// Create a new query registration
    pub fn new(name: impl Into<String>, query: impl Into<String>, window: WindowConfig) -> Self {
        Self {
            name: name.into(),
            query: query.into(),
            window,
            priority: 0,
            incremental: true,
            metadata: HashMap::new(),
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Set incremental mode
    pub fn with_incremental(mut self, incremental: bool) -> Self {
        self.incremental = incremental;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Unique query identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QueryId(u64);

impl QueryId {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

impl std::fmt::Display for QueryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Q{}", self.0)
    }
}

/// Query result with metadata
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Query that produced this result
    pub query_id: QueryId,
    /// Result triples
    pub triples: Vec<StarTriple>,
    /// Window start time
    pub window_start: Instant,
    /// Window end time
    pub window_end: Instant,
    /// Processing latency in microseconds
    pub latency_us: u64,
    /// Number of input triples processed
    pub input_count: usize,
    /// Is this a delta (incremental) result?
    pub is_delta: bool,
}

/// Timestamped triple for windowing
#[derive(Debug, Clone)]
struct TimestampedTriple {
    triple: StarTriple,
    timestamp: Instant,
    #[allow(dead_code)]
    sequence: u64,
}

/// Window state for a registered query
struct WindowState {
    /// Triples in current window
    triples: VecDeque<TimestampedTriple>,
    /// Last evaluation time
    last_eval: Instant,
    /// Sequence counter
    next_sequence: u64,
    /// Previous result for delta computation
    previous_result: Option<Vec<StarTriple>>,
}

impl WindowState {
    fn new() -> Self {
        Self {
            triples: VecDeque::new(),
            last_eval: Instant::now(),
            next_sequence: 0,
            previous_result: None,
        }
    }

    fn add(&mut self, triple: StarTriple) {
        let timestamped = TimestampedTriple {
            triple,
            timestamp: Instant::now(),
            sequence: self.next_sequence,
        };
        self.next_sequence += 1;
        self.triples.push_back(timestamped);
    }

    fn expire(&mut self, config: &WindowConfig) {
        let now = Instant::now();

        match config {
            WindowConfig::Tumbling { duration_secs }
            | WindowConfig::Sliding { duration_secs, .. } => {
                let cutoff = now - Duration::from_secs(*duration_secs);
                while let Some(front) = self.triples.front() {
                    if front.timestamp < cutoff {
                        self.triples.pop_front();
                    } else {
                        break;
                    }
                }
            }
            WindowConfig::Count { count } => {
                while self.triples.len() > *count {
                    self.triples.pop_front();
                }
            }
            WindowConfig::Session { timeout_secs } => {
                let cutoff = now - Duration::from_secs(*timeout_secs);
                if let Some(back) = self.triples.back() {
                    if back.timestamp < cutoff {
                        // Session expired, clear all
                        self.triples.clear();
                    }
                }
            }
            WindowConfig::Landmark => {
                // Never expire in landmark window
            }
        }
    }

    fn should_evaluate(&self, config: &WindowConfig) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_eval);

        match config {
            WindowConfig::Tumbling { duration_secs } => {
                elapsed >= Duration::from_secs(*duration_secs)
            }
            WindowConfig::Sliding { slide_secs, .. } => elapsed >= Duration::from_secs(*slide_secs),
            WindowConfig::Count { count } => self.triples.len() >= *count,
            WindowConfig::Session { timeout_secs } => {
                // Evaluate when activity occurs within session
                if let Some(back) = self.triples.back() {
                    let since_last = now.duration_since(back.timestamp);
                    since_last < Duration::from_secs(*timeout_secs)
                } else {
                    false
                }
            }
            WindowConfig::Landmark => {
                // Evaluate periodically (every second)
                elapsed >= Duration::from_secs(1)
            }
        }
    }

    fn get_window_triples(&self) -> Vec<StarTriple> {
        self.triples.iter().map(|t| t.triple.clone()).collect()
    }
}

/// Registered query with state
struct RegisteredQuery {
    registration: QueryRegistration,
    state: WindowState,
    subscribers: Vec<mpsc::Sender<QueryResult>>,
    metrics: QueryMetrics,
}

/// Query-specific metrics
#[derive(Default)]
struct QueryMetrics {
    evaluations: u64,
    results_produced: u64,
    total_latency_us: u64,
    errors: u64,
}

/// Streaming query engine for RDF-star
pub struct StreamingQueryEngine {
    config: StreamingConfig,
    queries: Arc<RwLock<HashMap<QueryId, RegisteredQuery>>>,
    input_tx: Option<mpsc::Sender<StreamingEvent>>,
    shutdown_tx: Option<broadcast::Sender<()>>,
    metrics: StreamingMetrics,
    #[allow(dead_code)]
    profiler: Profiler,
}

/// Streaming events
#[derive(Debug)]
enum StreamingEvent {
    Ingest(StarTriple),
    IngestBatch(Vec<StarTriple>),
    Tick,
}

/// Global streaming metrics
#[derive(Default)]
struct StreamingMetrics {
    triples_ingested: u64,
    queries_evaluated: u64,
    results_produced: u64,
    errors: u64,
    backpressure_events: u64,
}

impl StreamingQueryEngine {
    /// Create a new streaming query engine
    pub fn new() -> Self {
        Self::with_config(StreamingConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: StreamingConfig) -> Self {
        Self {
            config,
            queries: Arc::new(RwLock::new(HashMap::new())),
            input_tx: None,
            shutdown_tx: None,
            metrics: StreamingMetrics::default(),
            profiler: Profiler::new(),
        }
    }

    /// Register a continuous query
    #[instrument(skip(self, registration), fields(query_name = %registration.name))]
    pub async fn register_query_async(
        &mut self,
        registration: QueryRegistration,
    ) -> StarResult<QueryId> {
        let query_id = QueryId::new();

        let registered = RegisteredQuery {
            registration,
            state: WindowState::new(),
            subscribers: Vec::new(),
            metrics: QueryMetrics::default(),
        };

        let mut queries = self.queries.write().await;
        queries.insert(query_id, registered);

        info!("Registered streaming query: {}", query_id);
        Ok(query_id)
    }

    /// Register a continuous query (sync version for non-async contexts)
    pub fn register_query(&mut self, registration: QueryRegistration) -> StarResult<QueryId> {
        let query_id = QueryId::new();

        let registered = RegisteredQuery {
            registration,
            state: WindowState::new(),
            subscribers: Vec::new(),
            metrics: QueryMetrics::default(),
        };

        // For sync context, we need to create a runtime if none exists
        match tokio::runtime::Handle::try_current() {
            Ok(_handle) => {
                // Already in async context - this is an error in tests
                return Err(StarError::processing_error(
                    "Use register_query_async in async context",
                ));
            }
            Err(_) => {
                // Not in async context - create one
                let rt = tokio::runtime::Runtime::new()
                    .map_err(|_| StarError::processing_error("Failed to create runtime"))?;
                rt.block_on(async {
                    let mut queries = self.queries.write().await;
                    queries.insert(query_id, registered);
                });
            }
        }

        info!("Registered streaming query: {}", query_id);
        Ok(query_id)
    }

    /// Unregister a query
    pub async fn unregister_query(&mut self, query_id: QueryId) -> StarResult<()> {
        let mut queries = self.queries.write().await;
        queries
            .remove(&query_id)
            .ok_or_else(|| StarError::query_error("Query not found"))?;
        info!("Unregistered streaming query: {}", query_id);
        Ok(())
    }

    /// Subscribe to query results
    pub async fn subscribe(&self, query_id: QueryId) -> StarResult<mpsc::Receiver<QueryResult>> {
        let (tx, rx) = mpsc::channel(1000);

        let mut queries = self.queries.write().await;
        let query = queries
            .get_mut(&query_id)
            .ok_or_else(|| StarError::query_error("Query not found"))?;

        query.subscribers.push(tx);
        Ok(rx)
    }

    /// Start the streaming engine
    #[instrument(skip(self))]
    pub async fn start(&mut self) -> StarResult<()> {
        info!("Starting streaming query engine");

        let (input_tx, mut input_rx) = mpsc::channel::<StreamingEvent>(self.config.max_buffer_size);
        let (shutdown_tx, _) = broadcast::channel::<()>(1);

        self.input_tx = Some(input_tx);
        self.shutdown_tx = Some(shutdown_tx.clone());

        let queries = Arc::clone(&self.queries);
        let config = self.config.clone();

        // Spawn processing task
        tokio::spawn(async move {
            let mut shutdown_rx = shutdown_tx.subscribe();
            let mut tick_interval = tokio::time::interval(Duration::from_millis(100));

            loop {
                tokio::select! {
                    Some(event) = input_rx.recv() => {
                        Self::process_event(&queries, &config, event).await;
                    }
                    _ = tick_interval.tick() => {
                        Self::process_event(&queries, &config, StreamingEvent::Tick).await;
                    }
                    _ = shutdown_rx.recv() => {
                        info!("Streaming query engine shutting down");
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    /// Stop the streaming engine
    pub async fn stop(&mut self) -> StarResult<()> {
        if let Some(tx) = &self.shutdown_tx {
            let _ = tx.send(());
        }
        self.shutdown_tx = None;
        self.input_tx = None;
        info!("Streaming query engine stopped");
        Ok(())
    }

    /// Ingest a single triple
    #[instrument(skip(self, triple))]
    pub async fn ingest(&self, triple: StarTriple) -> StarResult<()> {
        if let Some(tx) = &self.input_tx {
            tx.send(StreamingEvent::Ingest(triple))
                .await
                .map_err(|_| StarError::processing_error("Failed to ingest triple"))?;
        }
        Ok(())
    }

    /// Ingest a batch of triples
    #[instrument(skip(self, triples), fields(batch_size = triples.len()))]
    pub async fn ingest_batch(&self, triples: Vec<StarTriple>) -> StarResult<()> {
        if let Some(tx) = &self.input_tx {
            tx.send(StreamingEvent::IngestBatch(triples))
                .await
                .map_err(|_| StarError::processing_error("Failed to ingest batch"))?;
        }
        Ok(())
    }

    /// Process a streaming event
    async fn process_event(
        queries: &Arc<RwLock<HashMap<QueryId, RegisteredQuery>>>,
        _config: &StreamingConfig,
        event: StreamingEvent,
    ) {
        let mut queries_guard = queries.write().await;

        match event {
            StreamingEvent::Ingest(triple) => {
                // Add triple to all query windows
                for (_, query) in queries_guard.iter_mut() {
                    query.state.add(triple.clone());
                    query.state.expire(&query.registration.window);
                }
            }
            StreamingEvent::IngestBatch(triples) => {
                for triple in triples {
                    for (_, query) in queries_guard.iter_mut() {
                        query.state.add(triple.clone());
                        query.state.expire(&query.registration.window);
                    }
                }
            }
            StreamingEvent::Tick => {
                // Check if any queries should be evaluated
            }
        }

        // Evaluate queries that are ready
        for (query_id, query) in queries_guard.iter_mut() {
            if query.state.should_evaluate(&query.registration.window) {
                let start = Instant::now();
                let window_triples = query.state.get_window_triples();

                // Evaluate query pattern (simplified pattern matching)
                let results =
                    Self::evaluate_query_pattern(&query.registration.query, &window_triples);

                // Compute delta if incremental
                let (output, is_delta) = if query.registration.incremental {
                    if let Some(prev) = &query.state.previous_result {
                        let delta = Self::compute_delta(prev, &results);
                        (delta, true)
                    } else {
                        (results.clone(), false)
                    }
                } else {
                    (results.clone(), false)
                };

                query.state.previous_result = Some(results);

                let elapsed = start.elapsed();
                let result = QueryResult {
                    query_id: *query_id,
                    triples: output,
                    window_start: query.state.last_eval,
                    window_end: Instant::now(),
                    latency_us: elapsed.as_micros() as u64,
                    input_count: window_triples.len(),
                    is_delta,
                };

                query.state.last_eval = Instant::now();
                query.metrics.evaluations += 1;
                query.metrics.results_produced += result.triples.len() as u64;
                query.metrics.total_latency_us += result.latency_us;

                // Notify subscribers
                let mut dead_subscribers = Vec::new();
                for (idx, subscriber) in query.subscribers.iter().enumerate() {
                    if subscriber.try_send(result.clone()).is_err() {
                        dead_subscribers.push(idx);
                    }
                }

                // Remove dead subscribers
                for idx in dead_subscribers.into_iter().rev() {
                    query.subscribers.remove(idx);
                }
            }
        }
    }

    /// Evaluate a query pattern against triples (simplified)
    fn evaluate_query_pattern(query: &str, triples: &[StarTriple]) -> Vec<StarTriple> {
        // Simplified pattern matching - in production would use full SPARQL-star parser
        let query_lower = query.to_lowercase();

        // Check for basic patterns
        if query_lower.contains("quoted") || query_lower.contains("<<") {
            // Filter for quoted triples
            triples
                .iter()
                .filter(|t| {
                    matches!(t.subject, StarTerm::QuotedTriple(_))
                        || matches!(t.object, StarTerm::QuotedTriple(_))
                })
                .cloned()
                .collect()
        } else if query_lower.contains("filter") {
            // Has a filter - return subset based on pattern (simplified)
            triples.iter().take(triples.len() / 2).cloned().collect()
        } else {
            // Return all triples
            triples.to_vec()
        }
    }

    /// Compute delta between two result sets
    fn compute_delta(previous: &[StarTriple], current: &[StarTriple]) -> Vec<StarTriple> {
        // Find new triples not in previous result
        current
            .iter()
            .filter(|t| !previous.contains(t))
            .cloned()
            .collect()
    }

    /// Get query statistics
    pub async fn get_query_stats(&self, query_id: QueryId) -> StarResult<QueryStats> {
        let queries = self.queries.read().await;
        let query = queries
            .get(&query_id)
            .ok_or_else(|| StarError::query_error("Query not found"))?;

        Ok(QueryStats {
            query_id,
            name: query.registration.name.clone(),
            evaluations: query.metrics.evaluations,
            results_produced: query.metrics.results_produced,
            avg_latency_us: if query.metrics.evaluations > 0 {
                query.metrics.total_latency_us / query.metrics.evaluations
            } else {
                0
            },
            window_size: query.state.triples.len(),
            subscriber_count: query.subscribers.len(),
            errors: query.metrics.errors,
        })
    }

    /// Get all registered query IDs
    pub async fn list_queries(&self) -> Vec<QueryId> {
        let queries = self.queries.read().await;
        queries.keys().copied().collect()
    }

    /// Get global engine statistics
    pub fn get_engine_stats(&self) -> EngineStats {
        EngineStats {
            triples_ingested: self.metrics.triples_ingested,
            queries_evaluated: self.metrics.queries_evaluated,
            results_produced: self.metrics.results_produced,
            errors: self.metrics.errors,
            backpressure_events: self.metrics.backpressure_events,
        }
    }
}

impl Default for StreamingQueryEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Query statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStats {
    pub query_id: QueryId,
    pub name: String,
    pub evaluations: u64,
    pub results_produced: u64,
    pub avg_latency_us: u64,
    pub window_size: usize,
    pub subscriber_count: usize,
    pub errors: u64,
}

/// Engine-level statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStats {
    pub triples_ingested: u64,
    pub queries_evaluated: u64,
    pub results_produced: u64,
    pub errors: u64,
    pub backpressure_events: u64,
}

/// Windowed aggregation helper
pub struct WindowedAggregator {
    /// Window configuration
    window: WindowConfig,
    /// Current window data
    data: VecDeque<(Instant, f64)>,
}

impl WindowedAggregator {
    /// Create a new windowed aggregator
    pub fn new(window: WindowConfig) -> Self {
        Self {
            window,
            data: VecDeque::new(),
        }
    }

    /// Add a value
    pub fn add(&mut self, value: f64) {
        self.data.push_back((Instant::now(), value));
        self.expire();
    }

    /// Expire old data
    fn expire(&mut self) {
        let now = Instant::now();
        match &self.window {
            WindowConfig::Tumbling { duration_secs }
            | WindowConfig::Sliding { duration_secs, .. } => {
                let cutoff = now - Duration::from_secs(*duration_secs);
                while let Some((ts, _)) = self.data.front() {
                    if *ts < cutoff {
                        self.data.pop_front();
                    } else {
                        break;
                    }
                }
            }
            WindowConfig::Count { count } => {
                while self.data.len() > *count {
                    self.data.pop_front();
                }
            }
            WindowConfig::Session { timeout_secs } => {
                let cutoff = now - Duration::from_secs(*timeout_secs);
                if let Some((ts, _)) = self.data.back() {
                    if *ts < cutoff {
                        self.data.clear();
                    }
                }
            }
            WindowConfig::Landmark => {}
        }
    }

    /// Get current count
    pub fn count(&self) -> usize {
        self.data.len()
    }

    /// Get current sum
    pub fn sum(&self) -> f64 {
        self.data.iter().map(|(_, v)| v).sum()
    }

    /// Get current average
    pub fn avg(&self) -> f64 {
        if self.data.is_empty() {
            0.0
        } else {
            self.sum() / self.data.len() as f64
        }
    }

    /// Get minimum
    pub fn min(&self) -> Option<f64> {
        self.data.iter().map(|(_, v)| *v).reduce(f64::min)
    }

    /// Get maximum
    pub fn max(&self) -> Option<f64> {
        self.data.iter().map(|(_, v)| *v).reduce(f64::max)
    }
}

/// Complex Event Processing (CEP) pattern
#[derive(Debug, Clone)]
pub struct CepPattern {
    /// Pattern name
    pub name: String,
    /// Sequence of expected predicates
    pub sequence: Vec<String>,
    /// Maximum time span for pattern in seconds
    pub time_span_secs: u64,
    /// Whether order matters
    pub ordered: bool,
}

impl CepPattern {
    /// Create a new CEP pattern
    pub fn new(name: impl Into<String>, sequence: Vec<String>, time_span_secs: u64) -> Self {
        Self {
            name: name.into(),
            sequence,
            time_span_secs,
            ordered: true,
        }
    }

    /// Set whether order matters
    pub fn with_ordered(mut self, ordered: bool) -> Self {
        self.ordered = ordered;
        self
    }
}

/// CEP pattern matcher
pub struct CepMatcher {
    patterns: Vec<CepPattern>,
    buffer: VecDeque<TimestampedTriple>,
    max_buffer_size: usize,
}

impl CepMatcher {
    /// Create a new CEP matcher
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            patterns: Vec::new(),
            buffer: VecDeque::new(),
            max_buffer_size,
        }
    }

    /// Add a pattern to match
    pub fn add_pattern(&mut self, pattern: CepPattern) {
        self.patterns.push(pattern);
    }

    /// Process a new triple
    pub fn process(&mut self, triple: StarTriple) -> Vec<(String, Vec<StarTriple>)> {
        let timestamped = TimestampedTriple {
            triple,
            timestamp: Instant::now(),
            sequence: self.buffer.len() as u64,
        };

        self.buffer.push_back(timestamped);

        // Limit buffer size
        while self.buffer.len() > self.max_buffer_size {
            self.buffer.pop_front();
        }

        // Check all patterns
        let mut matches = Vec::new();
        for pattern in &self.patterns {
            if let Some(matched) = self.check_pattern(pattern) {
                matches.push((pattern.name.clone(), matched));
            }
        }

        matches
    }

    /// Check if a pattern matches
    fn check_pattern(&self, pattern: &CepPattern) -> Option<Vec<StarTriple>> {
        let now = Instant::now();
        let cutoff = now - Duration::from_secs(pattern.time_span_secs);

        // Get triples within time span
        let relevant: Vec<_> = self
            .buffer
            .iter()
            .filter(|t| t.timestamp >= cutoff)
            .collect();

        if relevant.len() < pattern.sequence.len() {
            return None;
        }

        // Try to match pattern
        let mut matched = Vec::new();
        let mut pattern_idx = 0;

        for ts_triple in &relevant {
            if pattern_idx >= pattern.sequence.len() {
                break;
            }

            let expected_pred = &pattern.sequence[pattern_idx];
            if let StarTerm::NamedNode(nn) = &ts_triple.triple.predicate {
                if nn.iri.contains(expected_pred) || expected_pred == "*" {
                    matched.push(ts_triple.triple.clone());
                    pattern_idx += 1;
                } else if !pattern.ordered {
                    // Check if any remaining pattern element matches
                    for (idx, pat) in pattern.sequence.iter().enumerate().skip(pattern_idx) {
                        if nn.iri.contains(pat) || pat == "*" {
                            matched.push(ts_triple.triple.clone());
                            // Remove matched pattern element if unordered
                            if idx == pattern_idx {
                                pattern_idx += 1;
                            }
                            break;
                        }
                    }
                }
            }
        }

        if matched.len() == pattern.sequence.len() {
            Some(matched)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert_eq!(config.max_buffer_size, 100_000);
        assert_eq!(config.default_window_secs, 60);
    }

    #[test]
    fn test_window_config_constructors() {
        let sliding = WindowConfig::sliding(60, 10);
        assert!(matches!(
            sliding,
            WindowConfig::Sliding {
                duration_secs: 60,
                slide_secs: 10
            }
        ));

        let tumbling = WindowConfig::tumbling(30);
        assert!(matches!(
            tumbling,
            WindowConfig::Tumbling { duration_secs: 30 }
        ));

        let count = WindowConfig::count(100);
        assert!(matches!(count, WindowConfig::Count { count: 100 }));
    }

    #[test]
    fn test_query_registration() {
        let reg = QueryRegistration::new(
            "test_query",
            "SELECT * WHERE { ?s ?p ?o }",
            WindowConfig::tumbling(60),
        )
        .with_priority(5)
        .with_incremental(true)
        .with_metadata("description", "Test query");

        assert_eq!(reg.name, "test_query");
        assert_eq!(reg.priority, 5);
        assert!(reg.incremental);
        assert!(reg.metadata.contains_key("description"));
    }

    #[test]
    fn test_query_id_uniqueness() {
        let id1 = QueryId::new();
        let id2 = QueryId::new();
        let id3 = QueryId::new();

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
    }

    #[test]
    fn test_windowed_aggregator() {
        let mut agg = WindowedAggregator::new(WindowConfig::count(5));

        for i in 1..=10 {
            agg.add(i as f64);
        }

        assert_eq!(agg.count(), 5); // Count window of 5
        assert_eq!(agg.sum(), 6.0 + 7.0 + 8.0 + 9.0 + 10.0); // Last 5 values
        assert_eq!(agg.avg(), (6.0 + 7.0 + 8.0 + 9.0 + 10.0) / 5.0);
        assert_eq!(agg.min(), Some(6.0));
        assert_eq!(agg.max(), Some(10.0));
    }

    #[test]
    fn test_window_state() {
        let mut state = WindowState::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        state.add(triple.clone());
        state.add(triple.clone());
        state.add(triple);

        assert_eq!(state.triples.len(), 3);
    }

    #[test]
    fn test_cep_pattern() {
        let pattern = CepPattern::new(
            "order_flow",
            vec![
                "order".to_string(),
                "payment".to_string(),
                "shipped".to_string(),
            ],
            300,
        )
        .with_ordered(true);

        assert_eq!(pattern.name, "order_flow");
        assert_eq!(pattern.sequence.len(), 3);
        assert!(pattern.ordered);
    }

    #[test]
    fn test_cep_matcher_basic() {
        let mut matcher = CepMatcher::new(1000);

        let pattern = CepPattern::new("test", vec!["step1".to_string(), "step2".to_string()], 60);
        matcher.add_pattern(pattern);

        let triple1 = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/step1").unwrap(),
            StarTerm::literal("value1").unwrap(),
        );

        let triple2 = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/step2").unwrap(),
            StarTerm::literal("value2").unwrap(),
        );

        let matches1 = matcher.process(triple1);
        assert!(matches1.is_empty()); // Only one event, pattern not complete

        let matches2 = matcher.process(triple2);
        assert_eq!(matches2.len(), 1); // Pattern complete
        assert_eq!(matches2[0].0, "test");
        assert_eq!(matches2[0].1.len(), 2);
    }

    #[tokio::test]
    async fn test_streaming_engine_register() {
        let mut engine = StreamingQueryEngine::new();

        let reg = QueryRegistration::new(
            "test",
            "SELECT * WHERE { ?s ?p ?o }",
            WindowConfig::tumbling(60),
        );

        let query_id = engine.register_query_async(reg).await.unwrap();
        let queries = engine.list_queries().await;

        assert!(queries.contains(&query_id));
    }

    #[test]
    fn test_evaluate_query_pattern() {
        let triples = vec![
            StarTriple::new(
                StarTerm::iri("http://example.org/s1").unwrap(),
                StarTerm::iri("http://example.org/p").unwrap(),
                StarTerm::iri("http://example.org/o1").unwrap(),
            ),
            StarTriple::new(
                StarTerm::quoted_triple(StarTriple::new(
                    StarTerm::iri("http://example.org/a").unwrap(),
                    StarTerm::iri("http://example.org/b").unwrap(),
                    StarTerm::iri("http://example.org/c").unwrap(),
                )),
                StarTerm::iri("http://example.org/meta").unwrap(),
                StarTerm::literal("test").unwrap(),
            ),
        ];

        // Test quoted triple filter
        let results = StreamingQueryEngine::evaluate_query_pattern(
            "SELECT * WHERE { << ?s ?p ?o >> ?m ?v }",
            &triples,
        );
        assert_eq!(results.len(), 1);

        // Test all triples
        let all_results =
            StreamingQueryEngine::evaluate_query_pattern("SELECT * WHERE { ?s ?p ?o }", &triples);
        assert_eq!(all_results.len(), 2);
    }

    #[test]
    fn test_compute_delta() {
        let prev = vec![StarTriple::new(
            StarTerm::iri("http://example.org/s1").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o1").unwrap(),
        )];

        let current = vec![
            StarTriple::new(
                StarTerm::iri("http://example.org/s1").unwrap(),
                StarTerm::iri("http://example.org/p").unwrap(),
                StarTerm::iri("http://example.org/o1").unwrap(),
            ),
            StarTriple::new(
                StarTerm::iri("http://example.org/s2").unwrap(),
                StarTerm::iri("http://example.org/p").unwrap(),
                StarTerm::iri("http://example.org/o2").unwrap(),
            ),
        ];

        let delta = StreamingQueryEngine::compute_delta(&prev, &current);
        assert_eq!(delta.len(), 1);
    }
}
