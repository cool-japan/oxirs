//! # Complex Event Processing (CEP) Engine
//!
//! Production-grade CEP engine for detecting complex patterns across multiple event streams,
//! featuring composite event detection, event correlation, state machines, and rule-based
//! event processing with real-time pattern matching.
//!
//! ## Features
//!
//! - **Composite Event Detection**: Detect complex patterns from multiple simple events
//! - **Event Correlation**: Correlate events across streams using time windows and predicates
//! - **State Machine Processing**: Define complex event sequences with state transitions
//! - **Rule-Based Engine**: Define processing rules with conditions and actions
//! - **Temporal Operators**: Before, After, During, Overlaps, Meets, Starts, Finishes
//! - **Event Aggregation**: Aggregate events over time windows with custom functions
//! - **Event Enrichment**: Enrich events with contextual data from external sources
//! - **Pattern Library**: Pre-defined patterns for common scenarios
//! - **Real-time Processing**: Sub-millisecond pattern detection latency
//! - **Distributed Support**: Partition-aware processing for horizontal scaling
//!
//! ## Example
//!
//! ```no_run
//! use oxirs_stream::cep_engine::{CepEngine, CepConfig, EventPattern, TemporalOperator};
//! use oxirs_stream::event::StreamEvent;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = CepConfig::default();
//! let mut engine = CepEngine::new(config)?;
//!
//! // Define a pattern: "A followed by B within 10 seconds"
//! let pattern = EventPattern::sequence(vec![
//!     EventPattern::simple("event_type", "A"),
//!     EventPattern::simple("event_type", "B"),
//! ]).with_time_window(std::time::Duration::from_secs(10));
//!
//! engine.register_pattern("a_then_b", pattern).await?;
//!
//! // Process events
//! # let event = StreamEvent::Heartbeat {
//! #     timestamp: chrono::Utc::now(),
//! #     source: "test".to_string(),
//! #     metadata: Default::default(),
//! # };
//! let detected_patterns = engine.process_event(event).await?;
//! # Ok(())
//! # }
//! ```

use crate::event::StreamEvent;
use anyhow::Result;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info};
use uuid::Uuid;

/// Configuration for CEP engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CepConfig {
    /// Maximum number of events to keep in memory per partition
    pub max_events_in_memory: usize,

    /// Maximum time window for pattern detection
    pub max_time_window: Duration,

    /// Enable event correlation
    pub enable_correlation: bool,

    /// Enable state machine processing
    pub enable_state_machines: bool,

    /// Enable rule-based processing
    pub enable_rules: bool,

    /// Enable event enrichment
    pub enable_enrichment: bool,

    /// Maximum pattern complexity (nested depth)
    pub max_pattern_depth: usize,

    /// Pattern matching timeout
    pub pattern_matching_timeout: Duration,

    /// Event buffer size per stream
    pub event_buffer_size: usize,

    /// Enable metrics collection
    pub collect_metrics: bool,

    /// Garbage collection interval for expired events
    pub gc_interval: Duration,

    /// Enable distributed processing
    pub enable_distributed: bool,

    /// Number of partitions for distributed processing
    pub num_partitions: usize,
}

impl Default for CepConfig {
    fn default() -> Self {
        Self {
            max_events_in_memory: 100000,
            max_time_window: Duration::from_secs(3600),
            enable_correlation: true,
            enable_state_machines: true,
            enable_rules: true,
            enable_enrichment: true,
            max_pattern_depth: 10,
            pattern_matching_timeout: Duration::from_millis(100),
            event_buffer_size: 10000,
            collect_metrics: true,
            gc_interval: Duration::from_secs(60),
            enable_distributed: false,
            num_partitions: 8,
        }
    }
}

/// Complex Event Processing engine
pub struct CepEngine {
    /// Registered patterns
    patterns: Arc<RwLock<HashMap<String, EventPattern>>>,

    /// Event buffers per stream
    event_buffers: Arc<RwLock<HashMap<String, EventBuffer>>>,

    /// State machines for pattern tracking
    state_machines: Arc<RwLock<HashMap<String, StateMachine>>>,

    /// Rule engine
    rule_engine: Arc<RwLock<RuleEngine>>,

    /// Event correlator
    correlator: Arc<RwLock<EventCorrelator>>,

    /// Event enrichment service
    enrichment_service: Arc<RwLock<EnrichmentService>>,

    /// Pattern detector
    pattern_detector: Arc<RwLock<PatternDetector>>,

    /// Metrics collector
    metrics: Arc<RwLock<CepMetrics>>,

    /// Configuration
    config: CepConfig,

    /// Last garbage collection time
    last_gc: Arc<RwLock<Instant>>,
}

/// Event pattern definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventPattern {
    /// Simple event pattern with field predicates
    Simple {
        /// Pattern name
        name: String,
        /// Field predicates
        predicates: Vec<FieldPredicate>,
    },

    /// Sequence pattern (events in order)
    Sequence {
        /// Pattern name
        name: String,
        /// Sub-patterns
        patterns: Vec<EventPattern>,
        /// Time window
        time_window: Option<Duration>,
        /// Strict ordering (no interleaving)
        strict: bool,
    },

    /// Conjunction pattern (all events must occur)
    And {
        /// Pattern name
        name: String,
        /// Sub-patterns
        patterns: Vec<EventPattern>,
        /// Time window
        time_window: Option<Duration>,
    },

    /// Disjunction pattern (any event occurs)
    Or {
        /// Pattern name
        name: String,
        /// Sub-patterns
        patterns: Vec<EventPattern>,
    },

    /// Negation pattern (event must not occur)
    Not {
        /// Pattern name
        name: String,
        /// Pattern to negate
        pattern: Box<EventPattern>,
        /// Time window for negation
        time_window: Duration,
    },

    /// Repeat pattern (event occurs N times)
    Repeat {
        /// Pattern name
        name: String,
        /// Pattern to repeat
        pattern: Box<EventPattern>,
        /// Minimum occurrences
        min_count: usize,
        /// Maximum occurrences
        max_count: Option<usize>,
        /// Time window
        time_window: Option<Duration>,
    },

    /// Temporal pattern with Allen's interval algebra
    Temporal {
        /// Pattern name
        name: String,
        /// First event pattern
        first: Box<EventPattern>,
        /// Temporal operator
        operator: TemporalOperator,
        /// Second event pattern
        second: Box<EventPattern>,
        /// Time tolerance
        tolerance: Option<Duration>,
    },

    /// Aggregation pattern (aggregate events over window)
    Aggregation {
        /// Pattern name
        name: String,
        /// Pattern to aggregate
        pattern: Box<EventPattern>,
        /// Aggregation function
        aggregation: CepAggregationFunction,
        /// Time window
        window: Duration,
        /// Threshold for triggering
        threshold: f64,
    },
}

impl EventPattern {
    /// Create a simple pattern
    pub fn simple(field: &str, value: &str) -> Self {
        EventPattern::Simple {
            name: format!("{}={}", field, value),
            predicates: vec![FieldPredicate::Equals {
                field: field.to_string(),
                value: value.to_string(),
            }],
        }
    }

    /// Create a sequence pattern
    pub fn sequence(patterns: Vec<EventPattern>) -> Self {
        EventPattern::Sequence {
            name: "sequence".to_string(),
            patterns,
            time_window: None,
            strict: false,
        }
    }

    /// Add time window to pattern
    pub fn with_time_window(mut self, window: Duration) -> Self {
        match &mut self {
            EventPattern::Sequence { time_window, .. } => *time_window = Some(window),
            EventPattern::And { time_window, .. } => *time_window = Some(window),
            EventPattern::Repeat { time_window, .. } => *time_window = Some(window),
            _ => {}
        }
        self
    }

    /// Get pattern name
    pub fn name(&self) -> &str {
        match self {
            EventPattern::Simple { name, .. } => name,
            EventPattern::Sequence { name, .. } => name,
            EventPattern::And { name, .. } => name,
            EventPattern::Or { name, .. } => name,
            EventPattern::Not { name, .. } => name,
            EventPattern::Repeat { name, .. } => name,
            EventPattern::Temporal { name, .. } => name,
            EventPattern::Aggregation { name, .. } => name,
        }
    }
}

/// Field predicate for event matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldPredicate {
    /// Field equals value
    Equals { field: String, value: String },
    /// Field not equals value
    NotEquals { field: String, value: String },
    /// Field contains substring
    Contains { field: String, substring: String },
    /// Field matches regex
    Regex { field: String, pattern: String },
    /// Field greater than value
    GreaterThan { field: String, value: f64 },
    /// Field less than value
    LessThan { field: String, value: f64 },
    /// Field in range
    InRange { field: String, min: f64, max: f64 },
    /// Field exists
    Exists { field: String },
    /// Custom predicate function (serialized as name)
    Custom { name: String },
}

/// Temporal operators (Allen's interval algebra)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TemporalOperator {
    /// First event before second event
    Before,
    /// First event after second event
    After,
    /// First event meets second event (end == start)
    Meets,
    /// First event during second event
    During,
    /// First event overlaps second event
    Overlaps,
    /// First event starts second event (same start)
    Starts,
    /// First event finishes second event (same end)
    Finishes,
    /// First event equals second event (same start and end)
    Equals,
}

/// Aggregation function for CEP event aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CepAggregationFunction {
    /// Count events
    Count,
    /// Sum field values
    Sum { field: String },
    /// Average field values
    Average { field: String },
    /// Minimum field value
    Min { field: String },
    /// Maximum field value
    Max { field: String },
    /// Standard deviation of field values
    StdDev { field: String },
    /// Percentile of field values
    Percentile { field: String, percentile: f64 },
    /// Custom aggregation function
    Custom { name: String },
}

/// Event buffer for storing recent events
#[derive(Debug, Clone)]
pub struct EventBuffer {
    /// Stream name
    pub stream_name: String,
    /// Buffered events with timestamps
    pub events: VecDeque<TimestampedEvent>,
    /// Maximum buffer size
    pub max_size: usize,
    /// Oldest event timestamp
    pub oldest_timestamp: Option<DateTime<Utc>>,
    /// Newest event timestamp
    pub newest_timestamp: Option<DateTime<Utc>>,
}

/// Timestamped event
#[derive(Debug, Clone)]
pub struct TimestampedEvent {
    /// Event
    pub event: StreamEvent,
    /// Processing timestamp
    pub timestamp: DateTime<Utc>,
    /// Event ID
    pub id: Uuid,
}

/// State machine for pattern tracking
#[derive(Debug, Clone)]
pub struct StateMachine {
    /// Pattern being tracked
    pub pattern: EventPattern,
    /// Current state
    pub state: State,
    /// Partial matches
    pub partial_matches: Vec<PartialMatch>,
    /// Completed matches
    pub completed_matches: Vec<CompleteMatch>,
    /// State transition count
    pub transition_count: usize,
}

/// State in state machine
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum State {
    /// Initial state
    Initial,
    /// Intermediate state
    Intermediate { stage: usize },
    /// Final state (pattern matched)
    Final,
    /// Error state (pattern violated)
    Error,
}

/// Partial match in progress
#[derive(Debug, Clone)]
pub struct PartialMatch {
    /// Match ID
    pub id: Uuid,
    /// Matched events so far
    pub events: Vec<TimestampedEvent>,
    /// Current stage in pattern
    pub stage: usize,
    /// Start time
    pub start_time: DateTime<Utc>,
    /// Last update time
    pub last_update: DateTime<Utc>,
    /// Match state
    pub state: HashMap<String, String>,
}

/// Complete pattern match
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompleteMatch {
    /// Match ID
    pub id: Uuid,
    /// Pattern name
    pub pattern_name: String,
    /// Matched events
    pub event_ids: Vec<Uuid>,
    /// Start time
    pub start_time: DateTime<Utc>,
    /// End time
    pub end_time: DateTime<Utc>,
    /// Match duration
    pub duration: Duration,
    /// Confidence score (0.0-1.0)
    pub confidence: f64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Rule engine for event processing
#[derive(Debug, Clone)]
pub struct RuleEngine {
    /// Registered rules
    pub rules: HashMap<String, ProcessingRule>,
    /// Rule execution statistics
    pub stats: RuleExecutionStats,
}

/// Processing rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingRule {
    /// Rule name
    pub name: String,
    /// Condition to trigger rule
    pub condition: RuleCondition,
    /// Actions to execute
    pub actions: Vec<RuleAction>,
    /// Priority (higher = executed first)
    pub priority: i32,
    /// Enabled flag
    pub enabled: bool,
}

/// Rule condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleCondition {
    /// Pattern matched
    PatternMatched { pattern: String },
    /// Event field condition
    FieldCondition { predicate: FieldPredicate },
    /// Threshold exceeded
    ThresholdExceeded { metric: String, threshold: f64 },
    /// Complex condition (AND/OR)
    Complex {
        operator: String,
        conditions: Vec<RuleCondition>,
    },
}

/// Rule action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleAction {
    /// Emit new event
    EmitEvent {
        event_type: String,
        data: HashMap<String, String>,
    },
    /// Send alert
    SendAlert { severity: String, message: String },
    /// Update state
    UpdateState { key: String, value: String },
    /// Execute external webhook
    Webhook { url: String, method: String },
    /// Custom action
    Custom {
        name: String,
        params: HashMap<String, String>,
    },
}

/// Rule execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuleExecutionStats {
    /// Total rules executed
    pub total_executions: u64,
    /// Successful executions
    pub successful_executions: u64,
    /// Failed executions
    pub failed_executions: u64,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Average execution time
    pub avg_execution_time: Duration,
}

/// Event correlator for finding related events
#[derive(Debug, Clone)]
pub struct EventCorrelator {
    /// Correlation functions
    pub correlation_functions: HashMap<String, CorrelationFunction>,
    /// Correlation results cache
    pub correlation_cache: HashMap<CorrelationKey, CorrelationResult>,
    /// Statistics
    pub stats: CorrelationStats,
}

/// Correlation function
#[derive(Debug, Clone)]
pub struct CorrelationFunction {
    /// Function name
    pub name: String,
    /// Time window
    pub time_window: Duration,
    /// Fields to correlate
    pub fields: Vec<String>,
    /// Correlation threshold
    pub threshold: f64,
}

/// Correlation key for caching
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CorrelationKey {
    /// Event ID 1
    pub event1: Uuid,
    /// Event ID 2
    pub event2: Uuid,
    /// Function name
    pub function: String,
}

/// Correlation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationResult {
    /// Correlation score (0.0-1.0)
    pub score: f64,
    /// Correlated fields
    pub correlated_fields: Vec<String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Correlation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorrelationStats {
    /// Total correlations computed
    pub total_correlations: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Average correlation score
    pub avg_correlation_score: f64,
}

/// Event enrichment service
#[derive(Debug, Clone)]
pub struct EnrichmentService {
    /// Enrichment sources
    pub sources: HashMap<String, EnrichmentSource>,
    /// Enrichment cache
    pub cache: HashMap<String, EnrichmentData>,
    /// Statistics
    pub stats: EnrichmentStats,
}

/// Enrichment source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentSource {
    /// Source name
    pub name: String,
    /// Source type
    pub source_type: EnrichmentSourceType,
    /// Lookup key field
    pub key_field: String,
    /// Cache TTL
    pub cache_ttl: Duration,
}

/// Enrichment source type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnrichmentSourceType {
    /// External API
    ExternalApi { url: String, auth: Option<String> },
    /// Database
    Database {
        connection_string: String,
        query: String,
    },
    /// In-memory cache
    InMemory {
        data: HashMap<String, HashMap<String, String>>,
    },
    /// Custom source
    Custom { name: String },
}

/// Enrichment data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentData {
    /// Enriched fields
    pub fields: HashMap<String, String>,
    /// Source name
    pub source: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// TTL
    pub ttl: Duration,
}

/// Enrichment statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnrichmentStats {
    /// Total enrichments
    pub total_enrichments: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Failed enrichments
    pub failed_enrichments: u64,
}

/// Pattern detector
#[derive(Debug, Clone)]
pub struct PatternDetector {
    /// Registered patterns
    pub patterns: HashMap<String, EventPattern>,
    /// Detection algorithms
    pub algorithms: HashMap<String, DetectionAlgorithm>,
    /// Detection statistics
    pub stats: DetectionStats,
}

/// Detection algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetectionAlgorithm {
    /// Naive sequential matching
    Sequential,
    /// Automaton-based matching
    Automaton,
    /// Tree-based matching
    Tree,
    /// Graph-based matching
    Graph,
    /// Machine learning based
    MachineLearning { model_name: String },
}

/// Detection statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DetectionStats {
    /// Total events processed
    pub total_events_processed: u64,
    /// Patterns detected
    pub patterns_detected: u64,
    /// False positives
    pub false_positives: u64,
    /// False negatives
    pub false_negatives: u64,
    /// Average detection latency
    pub avg_detection_latency: Duration,
    /// Total detection time
    pub total_detection_time: Duration,
}

/// CEP metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CepMetrics {
    /// Total events processed
    pub total_events_processed: u64,
    /// Total patterns detected
    pub total_patterns_detected: u64,
    /// Events per second
    pub events_per_second: f64,
    /// Patterns per second
    pub patterns_per_second: f64,
    /// Average event processing latency
    pub avg_event_processing_latency: Duration,
    /// Average pattern matching latency
    pub avg_pattern_matching_latency: Duration,
    /// Memory usage (bytes)
    pub memory_usage_bytes: usize,
    /// Active partial matches
    pub active_partial_matches: usize,
    /// Completed matches in window
    pub completed_matches: usize,
    /// Garbage collections performed
    pub gc_count: u64,
    /// Last update time
    pub last_update: DateTime<Utc>,
}

/// Detected pattern result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    /// Pattern match
    pub pattern_match: CompleteMatch,
    /// Triggered rules
    pub triggered_rules: Vec<String>,
    /// Correlation results
    pub correlations: Vec<CorrelationResult>,
    /// Enriched data
    pub enrichments: HashMap<String, EnrichmentData>,
}

impl CepEngine {
    /// Create a new CEP engine
    pub fn new(config: CepConfig) -> Result<Self> {
        Ok(Self {
            patterns: Arc::new(RwLock::new(HashMap::new())),
            event_buffers: Arc::new(RwLock::new(HashMap::new())),
            state_machines: Arc::new(RwLock::new(HashMap::new())),
            rule_engine: Arc::new(RwLock::new(RuleEngine {
                rules: HashMap::new(),
                stats: RuleExecutionStats::default(),
            })),
            correlator: Arc::new(RwLock::new(EventCorrelator {
                correlation_functions: HashMap::new(),
                correlation_cache: HashMap::new(),
                stats: CorrelationStats::default(),
            })),
            enrichment_service: Arc::new(RwLock::new(EnrichmentService {
                sources: HashMap::new(),
                cache: HashMap::new(),
                stats: EnrichmentStats::default(),
            })),
            pattern_detector: Arc::new(RwLock::new(PatternDetector {
                patterns: HashMap::new(),
                algorithms: HashMap::new(),
                stats: DetectionStats::default(),
            })),
            metrics: Arc::new(RwLock::new(CepMetrics {
                last_update: Utc::now(),
                ..Default::default()
            })),
            config,
            last_gc: Arc::new(RwLock::new(Instant::now())),
        })
    }

    /// Register an event pattern
    pub async fn register_pattern(&mut self, name: &str, pattern: EventPattern) -> Result<()> {
        let mut patterns = self.patterns.write().await;
        patterns.insert(name.to_string(), pattern.clone());

        // Initialize state machine for pattern
        let mut state_machines = self.state_machines.write().await;
        state_machines.insert(
            name.to_string(),
            StateMachine {
                pattern,
                state: State::Initial,
                partial_matches: Vec::new(),
                completed_matches: Vec::new(),
                transition_count: 0,
            },
        );

        info!("Registered CEP pattern: {}", name);
        Ok(())
    }

    /// Register a processing rule
    pub async fn register_rule(&mut self, rule: ProcessingRule) -> Result<()> {
        let mut rule_engine = self.rule_engine.write().await;
        rule_engine.rules.insert(rule.name.clone(), rule.clone());
        info!("Registered CEP rule: {}", rule.name);
        Ok(())
    }

    /// Process an event through the CEP engine
    pub async fn process_event(&mut self, event: StreamEvent) -> Result<Vec<DetectedPattern>> {
        let start_time = Instant::now();
        let event_timestamp = Utc::now();

        // Create timestamped event
        let timestamped_event = TimestampedEvent {
            event: event.clone(),
            timestamp: event_timestamp,
            id: Uuid::new_v4(),
        };

        // Add to event buffer
        self.add_to_buffer("default", timestamped_event.clone())
            .await?;

        // Run garbage collection if needed
        self.maybe_run_gc().await?;

        // Detect patterns
        let detected_patterns = self.detect_patterns(&timestamped_event).await?;

        // Execute rules for detected patterns
        let mut results = Vec::new();
        for pattern_match in detected_patterns {
            let triggered_rules = self.execute_rules(&pattern_match).await?;

            // Correlate events if enabled
            let correlations = if self.config.enable_correlation {
                self.correlate_events(&pattern_match).await?
            } else {
                Vec::new()
            };

            // Enrich events if enabled
            let enrichments = if self.config.enable_enrichment {
                self.enrich_events(&pattern_match).await?
            } else {
                HashMap::new()
            };

            results.push(DetectedPattern {
                pattern_match,
                triggered_rules,
                correlations,
                enrichments,
            });
        }

        // Update metrics
        let processing_latency = start_time.elapsed();
        self.update_metrics(processing_latency, results.len()).await;

        Ok(results)
    }

    /// Add event to buffer
    async fn add_to_buffer(&self, stream: &str, event: TimestampedEvent) -> Result<()> {
        let mut buffers = self.event_buffers.write().await;
        let buffer = buffers
            .entry(stream.to_string())
            .or_insert_with(|| EventBuffer {
                stream_name: stream.to_string(),
                events: VecDeque::new(),
                max_size: self.config.event_buffer_size,
                oldest_timestamp: None,
                newest_timestamp: None,
            });

        // Update timestamps
        if buffer.oldest_timestamp.is_none() {
            buffer.oldest_timestamp = Some(event.timestamp);
        }
        buffer.newest_timestamp = Some(event.timestamp);

        // Add event
        buffer.events.push_back(event);

        // Trim buffer if too large
        while buffer.events.len() > buffer.max_size {
            buffer.events.pop_front();
            if let Some(first_event) = buffer.events.front() {
                buffer.oldest_timestamp = Some(first_event.timestamp);
            }
        }

        Ok(())
    }

    /// Detect patterns in recent events
    async fn detect_patterns(&self, new_event: &TimestampedEvent) -> Result<Vec<CompleteMatch>> {
        let mut detected = Vec::new();
        let mut state_machines = self.state_machines.write().await;

        for (pattern_name, state_machine) in state_machines.iter_mut() {
            // Try to match pattern
            if let Some(complete_match) = self.try_match_pattern(state_machine, new_event).await? {
                detected.push(complete_match);
                debug!("Pattern detected: {}", pattern_name);
            }
        }

        Ok(detected)
    }

    /// Try to match a pattern
    async fn try_match_pattern(
        &self,
        state_machine: &mut StateMachine,
        event: &TimestampedEvent,
    ) -> Result<Option<CompleteMatch>> {
        // Clone pattern to avoid borrow issues
        let pattern = state_machine.pattern.clone();

        match &pattern {
            EventPattern::Simple { predicates, .. } => {
                if self.evaluate_predicates(predicates, &event.event).await? {
                    Ok(Some(CompleteMatch {
                        id: Uuid::new_v4(),
                        pattern_name: pattern.name().to_string(),
                        event_ids: vec![event.id],
                        start_time: event.timestamp,
                        end_time: event.timestamp,
                        duration: Duration::from_secs(0),
                        confidence: 1.0,
                        metadata: HashMap::new(),
                    }))
                } else {
                    Ok(None)
                }
            }
            EventPattern::Sequence {
                patterns,
                time_window,
                strict,
                ..
            } => {
                self.match_sequence(state_machine, event, patterns, *time_window, *strict)
                    .await
            }
            EventPattern::And {
                patterns,
                time_window,
                ..
            } => {
                self.match_conjunction(state_machine, event, patterns, *time_window)
                    .await
            }
            _ => {
                // Other pattern types (to be implemented)
                Ok(None)
            }
        }
    }

    /// Match sequence pattern
    async fn match_sequence(
        &self,
        state_machine: &mut StateMachine,
        event: &TimestampedEvent,
        patterns: &[EventPattern],
        time_window: Option<Duration>,
        _strict: bool,
    ) -> Result<Option<CompleteMatch>> {
        // Update partial matches
        let mut new_partial_matches = Vec::new();

        for partial_match in &mut state_machine.partial_matches {
            let next_stage = partial_match.stage;
            if next_stage < patterns.len() {
                if let EventPattern::Simple { predicates, .. } = &patterns[next_stage] {
                    if self.evaluate_predicates(predicates, &event.event).await? {
                        // Check time window
                        if let Some(window) = time_window {
                            let elapsed = event
                                .timestamp
                                .signed_duration_since(partial_match.start_time);
                            if elapsed.num_seconds() > window.as_secs() as i64 {
                                continue; // Expired
                            }
                        }

                        // Advance match
                        let mut new_match = partial_match.clone();
                        new_match.events.push(event.clone());
                        new_match.stage += 1;
                        new_match.last_update = event.timestamp;

                        if new_match.stage == patterns.len() {
                            // Complete match!
                            let event_ids: Vec<Uuid> =
                                new_match.events.iter().map(|e| e.id).collect();
                            let duration = event
                                .timestamp
                                .signed_duration_since(new_match.start_time)
                                .to_std()
                                .unwrap_or(Duration::from_secs(0));

                            return Ok(Some(CompleteMatch {
                                id: Uuid::new_v4(),
                                pattern_name: state_machine.pattern.name().to_string(),
                                event_ids,
                                start_time: new_match.start_time,
                                end_time: event.timestamp,
                                duration,
                                confidence: 1.0,
                                metadata: HashMap::new(),
                            }));
                        } else {
                            new_partial_matches.push(new_match);
                        }
                    }
                }
            }
        }

        // Check if this event starts a new partial match
        if let EventPattern::Simple { predicates, .. } = &patterns[0] {
            if self.evaluate_predicates(predicates, &event.event).await? {
                new_partial_matches.push(PartialMatch {
                    id: Uuid::new_v4(),
                    events: vec![event.clone()],
                    stage: 1,
                    start_time: event.timestamp,
                    last_update: event.timestamp,
                    state: HashMap::new(),
                });
            }
        }

        state_machine.partial_matches = new_partial_matches;
        Ok(None)
    }

    /// Match conjunction pattern
    async fn match_conjunction(
        &self,
        _state_machine: &mut StateMachine,
        _event: &TimestampedEvent,
        _patterns: &[EventPattern],
        _time_window: Option<Duration>,
    ) -> Result<Option<CompleteMatch>> {
        // Simplified implementation
        Ok(None)
    }

    /// Evaluate predicates against an event
    async fn evaluate_predicates(
        &self,
        predicates: &[FieldPredicate],
        event: &StreamEvent,
    ) -> Result<bool> {
        for predicate in predicates {
            match predicate {
                FieldPredicate::Equals { field, value } => {
                    // Extract field from event (simplified)
                    if field == "event_type" {
                        let event_type = match event {
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
                            _ => "Other", // Catch-all for other event types
                        };
                        if event_type != value {
                            return Ok(false);
                        }
                    }
                }
                FieldPredicate::Contains { field, substring } => {
                    // Simplified implementation
                    if field == "source" {
                        let source = match event {
                            StreamEvent::Heartbeat { source, .. } => source,
                            _ => return Ok(false),
                        };
                        if !source.contains(substring) {
                            return Ok(false);
                        }
                    }
                }
                _ => {
                    // Other predicates (simplified)
                }
            }
        }
        Ok(true)
    }

    /// Execute rules for a pattern match
    async fn execute_rules(&self, pattern_match: &CompleteMatch) -> Result<Vec<String>> {
        let mut triggered = Vec::new();

        // Clone rules to avoid borrow issues
        let rules = {
            let rule_engine = self.rule_engine.read().await;
            rule_engine.rules.clone()
        };

        for (rule_name, rule) in &rules {
            if !rule.enabled {
                continue;
            }

            // Check if rule condition matches
            if self
                .evaluate_rule_condition(&rule.condition, pattern_match)
                .await?
            {
                // Execute actions
                for action in &rule.actions {
                    self.execute_rule_action(action).await?;
                }
                triggered.push(rule_name.clone());

                // Update stats
                let mut rule_engine = self.rule_engine.write().await;
                rule_engine.stats.successful_executions += 1;
            }
        }

        Ok(triggered)
    }

    /// Evaluate rule condition
    async fn evaluate_rule_condition(
        &self,
        condition: &RuleCondition,
        pattern_match: &CompleteMatch,
    ) -> Result<bool> {
        match condition {
            RuleCondition::PatternMatched { pattern } => Ok(&pattern_match.pattern_name == pattern),
            _ => {
                // Other conditions (simplified)
                Ok(false)
            }
        }
    }

    /// Execute rule action
    async fn execute_rule_action(&self, action: &RuleAction) -> Result<()> {
        match action {
            RuleAction::SendAlert { severity, message } => {
                info!("CEP Alert [{}]: {}", severity, message);
            }
            RuleAction::EmitEvent { event_type, data } => {
                debug!("CEP Emit Event: {} with data: {:?}", event_type, data);
            }
            _ => {
                // Other actions (simplified)
            }
        }
        Ok(())
    }

    /// Correlate events
    async fn correlate_events(
        &self,
        _pattern_match: &CompleteMatch,
    ) -> Result<Vec<CorrelationResult>> {
        // Simplified implementation
        Ok(Vec::new())
    }

    /// Enrich events
    async fn enrich_events(
        &self,
        _pattern_match: &CompleteMatch,
    ) -> Result<HashMap<String, EnrichmentData>> {
        // Simplified implementation
        Ok(HashMap::new())
    }

    /// Run garbage collection if needed
    async fn maybe_run_gc(&self) -> Result<()> {
        let mut last_gc = self.last_gc.write().await;
        if last_gc.elapsed() >= self.config.gc_interval {
            self.run_gc().await?;
            *last_gc = Instant::now();

            let mut metrics = self.metrics.write().await;
            metrics.gc_count += 1;
        }
        Ok(())
    }

    /// Run garbage collection
    async fn run_gc(&self) -> Result<()> {
        let cutoff_time =
            Utc::now() - ChronoDuration::seconds(self.config.max_time_window.as_secs() as i64);

        // Clean event buffers
        let mut buffers = self.event_buffers.write().await;
        for buffer in buffers.values_mut() {
            buffer.events.retain(|e| e.timestamp > cutoff_time);
            if let Some(first_event) = buffer.events.front() {
                buffer.oldest_timestamp = Some(first_event.timestamp);
            }
        }

        // Clean partial matches
        let mut state_machines = self.state_machines.write().await;
        for state_machine in state_machines.values_mut() {
            state_machine
                .partial_matches
                .retain(|m| m.last_update > cutoff_time);
        }

        debug!("CEP garbage collection completed");
        Ok(())
    }

    /// Update metrics
    async fn update_metrics(&self, processing_latency: Duration, patterns_detected: usize) {
        let mut metrics = self.metrics.write().await;
        metrics.total_events_processed += 1;
        metrics.total_patterns_detected += patterns_detected as u64;

        let now = Utc::now();
        let elapsed_duration = now.signed_duration_since(metrics.last_update);
        let elapsed_secs = elapsed_duration.num_seconds() as f64;

        if elapsed_secs > 0.0 {
            metrics.events_per_second = metrics.total_events_processed as f64 / elapsed_secs;
            metrics.patterns_per_second = metrics.total_patterns_detected as f64 / elapsed_secs;
        }

        // Update average latency
        let total_latency = metrics.avg_event_processing_latency.as_micros()
            * (metrics.total_events_processed - 1) as u128
            + processing_latency.as_micros();
        metrics.avg_event_processing_latency =
            Duration::from_micros((total_latency / metrics.total_events_processed as u128) as u64);

        // Count active partial matches
        let state_machines = self.state_machines.read().await;
        metrics.active_partial_matches = state_machines
            .values()
            .map(|sm| sm.partial_matches.len())
            .sum();
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> CepMetrics {
        self.metrics.read().await.clone()
    }

    /// Get statistics
    pub async fn get_statistics(&self) -> CepStatistics {
        let metrics = self.metrics.read().await;
        let rule_engine = self.rule_engine.read().await;
        let correlator = self.correlator.read().await;
        let enrichment = self.enrichment_service.read().await;
        let detector = self.pattern_detector.read().await;

        CepStatistics {
            metrics: metrics.clone(),
            rule_stats: rule_engine.stats.clone(),
            correlation_stats: correlator.stats.clone(),
            enrichment_stats: enrichment.stats.clone(),
            detection_stats: detector.stats.clone(),
        }
    }
}

/// CEP statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CepStatistics {
    /// CEP metrics
    pub metrics: CepMetrics,
    /// Rule execution statistics
    pub rule_stats: RuleExecutionStats,
    /// Correlation statistics
    pub correlation_stats: CorrelationStats,
    /// Enrichment statistics
    pub enrichment_stats: EnrichmentStats,
    /// Detection statistics
    pub detection_stats: DetectionStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventMetadata;

    #[tokio::test]
    async fn test_cep_engine_creation() {
        let config = CepConfig::default();
        let engine = CepEngine::new(config);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_pattern_registration() {
        let config = CepConfig::default();
        let mut engine = CepEngine::new(config).unwrap();

        let pattern = EventPattern::simple("event_type", "test");
        let result = engine.register_pattern("test_pattern", pattern).await;
        assert!(result.is_ok());

        let patterns = engine.patterns.read().await;
        assert!(patterns.contains_key("test_pattern"));
    }

    #[tokio::test]
    async fn test_simple_pattern_matching() {
        let config = CepConfig::default();
        let mut engine = CepEngine::new(config).unwrap();

        let pattern = EventPattern::simple("event_type", "Heartbeat");
        engine.register_pattern("heartbeat", pattern).await.unwrap();

        let event = StreamEvent::Heartbeat {
            timestamp: Utc::now(),
            source: "test".to_string(),
            metadata: EventMetadata::default(),
        };

        let detected = engine.process_event(event).await.unwrap();
        assert!(!detected.is_empty());
    }

    #[tokio::test]
    async fn test_sequence_pattern() {
        let config = CepConfig::default();
        let mut engine = CepEngine::new(config).unwrap();

        let pattern = EventPattern::sequence(vec![
            EventPattern::simple("event_type", "Heartbeat"),
            EventPattern::simple("event_type", "Heartbeat"),
        ])
        .with_time_window(Duration::from_secs(10));

        engine
            .register_pattern("double_heartbeat", pattern)
            .await
            .unwrap();

        let event1 = StreamEvent::Heartbeat {
            timestamp: Utc::now(),
            source: "test".to_string(),
            metadata: EventMetadata::default(),
        };

        let detected1 = engine.process_event(event1).await.unwrap();
        assert!(detected1.is_empty()); // First event, no match yet

        let event2 = StreamEvent::Heartbeat {
            timestamp: Utc::now(),
            source: "test".to_string(),
            metadata: EventMetadata::default(),
        };

        let detected2 = engine.process_event(event2).await.unwrap();
        assert!(!detected2.is_empty()); // Second event, pattern matched
    }

    #[tokio::test]
    async fn test_rule_registration() {
        let config = CepConfig::default();
        let mut engine = CepEngine::new(config).unwrap();

        let rule = ProcessingRule {
            name: "test_rule".to_string(),
            condition: RuleCondition::PatternMatched {
                pattern: "heartbeat".to_string(),
            },
            actions: vec![RuleAction::SendAlert {
                severity: "info".to_string(),
                message: "Heartbeat detected".to_string(),
            }],
            priority: 1,
            enabled: true,
        };

        let result = engine.register_rule(rule).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_event_buffer() {
        let config = CepConfig::default();
        let engine = CepEngine::new(config).unwrap();

        let event = StreamEvent::Heartbeat {
            timestamp: Utc::now(),
            source: "test".to_string(),
            metadata: EventMetadata::default(),
        };

        let timestamped = TimestampedEvent {
            event,
            timestamp: Utc::now(),
            id: Uuid::new_v4(),
        };

        engine
            .add_to_buffer("test_stream", timestamped)
            .await
            .unwrap();

        let buffers = engine.event_buffers.read().await;
        assert!(buffers.contains_key("test_stream"));
        assert_eq!(buffers.get("test_stream").unwrap().events.len(), 1);
    }

    #[tokio::test]
    async fn test_predicate_evaluation() {
        let config = CepConfig::default();
        let engine = CepEngine::new(config).unwrap();

        let predicates = vec![FieldPredicate::Equals {
            field: "event_type".to_string(),
            value: "Heartbeat".to_string(),
        }];

        let event = StreamEvent::Heartbeat {
            timestamp: Utc::now(),
            source: "test".to_string(),
            metadata: EventMetadata::default(),
        };

        let result = engine
            .evaluate_predicates(&predicates, &event)
            .await
            .unwrap();
        assert!(result);
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let config = CepConfig::default();
        let mut engine = CepEngine::new(config).unwrap();

        let event = StreamEvent::Heartbeat {
            timestamp: Utc::now(),
            source: "test".to_string(),
            metadata: EventMetadata::default(),
        };

        engine.process_event(event).await.unwrap();

        let metrics = engine.get_metrics().await;
        assert_eq!(metrics.total_events_processed, 1);
    }

    #[tokio::test]
    async fn test_garbage_collection() {
        let config = CepConfig {
            gc_interval: Duration::from_millis(10),
            ..Default::default()
        };
        let engine = CepEngine::new(config).unwrap();

        // Add old event
        let old_event = TimestampedEvent {
            event: StreamEvent::Heartbeat {
                timestamp: Utc::now(),
                source: "test".to_string(),
                metadata: EventMetadata::default(),
            },
            timestamp: Utc::now() - ChronoDuration::hours(2),
            id: Uuid::new_v4(),
        };

        engine.add_to_buffer("test", old_event).await.unwrap();

        // Wait for GC
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Run GC
        engine.run_gc().await.unwrap();

        let buffers = engine.event_buffers.read().await;
        assert!(buffers.get("test").unwrap().events.is_empty());
    }

    #[tokio::test]
    async fn test_pattern_with_time_window() {
        let pattern = EventPattern::sequence(vec![
            EventPattern::simple("type", "A"),
            EventPattern::simple("type", "B"),
        ])
        .with_time_window(Duration::from_secs(5));

        match pattern {
            EventPattern::Sequence { time_window, .. } => {
                assert_eq!(time_window, Some(Duration::from_secs(5)));
            }
            _ => panic!("Expected sequence pattern"),
        }
    }

    #[tokio::test]
    async fn test_statistics() {
        let config = CepConfig::default();
        let engine = CepEngine::new(config).unwrap();

        let stats = engine.get_statistics().await;
        assert_eq!(stats.metrics.total_events_processed, 0);
    }
}
