//! Type definitions for federated query execution
//!
//! This module contains all the type definitions, configuration structs, and data structures
//! used throughout the federated execution system.

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{
    cache::{CacheConfig, FederationCache},
    planner::StepType,
    service_executor::{JoinExecutor, ServiceExecutor, ServiceExecutorConfig},
};

/// Federated query executor
#[derive(Debug)]
pub struct FederatedExecutor {
    pub client: Client,
    pub config: FederatedExecutorConfig,
    pub service_executor: Arc<ServiceExecutor>,
    pub join_executor: Arc<JoinExecutor>,
    pub cache: Arc<FederationCache>,
}

/// Configuration for the federated executor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedExecutorConfig {
    pub request_timeout: Duration,
    pub max_parallel_requests: usize,
    pub user_agent: String,
    pub cache_config: CacheConfig,
    pub service_executor_config: ServiceExecutorConfig,
    pub enable_adaptive_execution: bool,
    pub enable_performance_monitoring: bool,
    pub enable_circuit_breaker: bool,
    pub circuit_breaker_config: CircuitBreakerConfig,
    pub retry_config: RetryConfig,
}

impl Default for FederatedExecutorConfig {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(30),
            max_parallel_requests: 10,
            user_agent: "oxirs-federate/1.0".to_string(),
            cache_config: CacheConfig::default(),
            service_executor_config: ServiceExecutorConfig::default(),
            enable_adaptive_execution: true,
            enable_performance_monitoring: true,
            enable_circuit_breaker: true,
            circuit_breaker_config: CircuitBreakerConfig::default(),
            retry_config: RetryConfig::default(),
        }
    }
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: usize,
    pub recovery_timeout: Duration,
    pub half_open_max_calls: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(60),
            half_open_max_calls: 3,
        }
    }
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: usize,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
        }
    }
}

/// Result of executing a single step in the execution plan
#[derive(Debug, Clone)]
pub struct StepResult {
    pub step_id: String,
    pub step_type: StepType,
    pub status: ExecutionStatus,
    pub data: Option<QueryResultData>,
    pub error: Option<String>,
    pub execution_time: Duration,
    pub service_id: Option<String>,
    pub memory_used: usize,
    pub result_size: usize,
    pub success: bool,
    pub error_message: Option<String>,
    pub service_response_time: Duration,
    pub cache_hit: bool,
}

/// Status of step execution
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionStatus {
    Success,
    Failed,
    Timeout,
    Cancelled,
}

/// SPARQL query results structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlResults {
    pub head: SparqlHead,
    pub results: SparqlResultsData,
}

/// SPARQL results head
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlHead {
    pub vars: Vec<String>,
}

/// SPARQL results data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlResultsData {
    pub bindings: Vec<SparqlBinding>,
}

impl SparqlResultsData {
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

/// A single SPARQL binding (variable -> value mapping)
pub type SparqlBinding = HashMap<String, SparqlValue>;

/// A SPARQL value with RDF-star support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlValue {
    #[serde(rename = "type")]
    pub value_type: String,
    pub value: String,
    pub datatype: Option<String>,
    pub lang: Option<String>,
    /// Additional quoted triple data for RDF-star support
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quoted_triple: Option<QuotedTripleValue>,
}

/// Quoted triple value for RDF-star support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotedTripleValue {
    pub subject: Box<RdfTerm>,
    pub predicate: Box<RdfTerm>,
    pub object: Box<RdfTerm>,
}

/// RDF term representation for quoted triples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RdfTerm {
    /// IRI reference
    IRI(String),
    /// Blank node
    BlankNode(String),
    /// Literal value
    Literal {
        value: String,
        datatype: Option<String>,
        lang: Option<String>,
    },
    /// Nested quoted triple
    QuotedTriple(QuotedTripleValue),
}

impl SparqlValue {
    /// Create a new literal SPARQL value
    pub fn literal(value: String, datatype: Option<String>, lang: Option<String>) -> Self {
        Self {
            value_type: "literal".to_string(),
            value,
            datatype,
            lang,
            quoted_triple: None,
        }
    }

    /// Create a new IRI SPARQL value
    pub fn iri(iri: String) -> Self {
        Self {
            value_type: "uri".to_string(),
            value: iri,
            datatype: None,
            lang: None,
            quoted_triple: None,
        }
    }

    /// Create a new blank node SPARQL value
    pub fn blank_node(id: String) -> Self {
        Self {
            value_type: "bnode".to_string(),
            value: id,
            datatype: None,
            lang: None,
            quoted_triple: None,
        }
    }

    /// Create a new quoted triple SPARQL value for RDF-star
    pub fn quoted_triple(subject: RdfTerm, predicate: RdfTerm, object: RdfTerm) -> Self {
        let quoted_triple = QuotedTripleValue {
            subject: Box::new(subject.clone()),
            predicate: Box::new(predicate.clone()),
            object: Box::new(object.clone()),
        };

        Self {
            value_type: "quoted_triple".to_string(),
            value: format!(
                "<<{} {} {}>>",
                quoted_triple.subject, quoted_triple.predicate, quoted_triple.object
            ),
            datatype: None,
            lang: None,
            quoted_triple: Some(quoted_triple),
        }
    }

    /// Check if this value represents a quoted triple
    pub fn is_quoted_triple(&self) -> bool {
        self.quoted_triple.is_some()
    }

    /// Get the quoted triple data if available
    pub fn get_quoted_triple(&self) -> Option<&QuotedTripleValue> {
        self.quoted_triple.as_ref()
    }

    /// Convert to compact encoded format for storage
    pub fn to_encoded_format(&self) -> EncodedSparqlValue {
        match &self.quoted_triple {
            Some(qt) => EncodedSparqlValue::QuotedTriple {
                subject: qt.subject.to_encoded_term(),
                predicate: qt.predicate.to_encoded_term(),
                object: qt.object.to_encoded_term(),
            },
            None => match self.value_type.as_str() {
                "uri" => EncodedSparqlValue::IRI(self.value.clone()),
                "bnode" => EncodedSparqlValue::BlankNode(self.value.clone()),
                "literal" => EncodedSparqlValue::Literal {
                    value: self.value.clone(),
                    datatype: self.datatype.clone(),
                    lang: self.lang.clone(),
                },
                _ => EncodedSparqlValue::Unknown(self.value.clone()),
            },
        }
    }
}

impl fmt::Display for RdfTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RdfTerm::IRI(iri) => write!(f, "<{iri}>"),
            RdfTerm::BlankNode(id) => write!(f, "_:{id}"),
            RdfTerm::Literal {
                value,
                datatype,
                lang,
            } => {
                write!(f, "\"{value}\"")?;
                if let Some(lang) = lang {
                    write!(f, "@{lang}")
                } else if let Some(datatype) = datatype {
                    write!(f, "^^<{datatype}>")
                } else {
                    Ok(())
                }
            }
            RdfTerm::QuotedTriple(qt) => {
                write!(f, "<<{} {} {}>>", qt.subject, qt.predicate, qt.object)
            }
        }
    }
}

impl RdfTerm {
    /// Convert to encoded term for storage optimization
    pub fn to_encoded_term(&self) -> EncodedTerm {
        match self {
            RdfTerm::IRI(iri) => EncodedTerm::IRI(iri.clone()),
            RdfTerm::BlankNode(id) => EncodedTerm::BlankNode(id.clone()),
            RdfTerm::Literal {
                value,
                datatype,
                lang,
            } => EncodedTerm::Literal {
                value: value.clone(),
                datatype: datatype.clone(),
                lang: lang.clone(),
            },
            RdfTerm::QuotedTriple(qt) => EncodedTerm::QuotedTriple {
                subject: Box::new(qt.subject.to_encoded_term()),
                predicate: Box::new(qt.predicate.to_encoded_term()),
                object: Box::new(qt.object.to_encoded_term()),
            },
        }
    }
}

/// Compact encoded format for efficient storage of SPARQL values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncodedSparqlValue {
    IRI(String),
    BlankNode(String),
    Literal {
        value: String,
        datatype: Option<String>,
        lang: Option<String>,
    },
    QuotedTriple {
        subject: EncodedTerm,
        predicate: EncodedTerm,
        object: EncodedTerm,
    },
    Unknown(String),
}

/// Compact encoded format for RDF terms in storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncodedTerm {
    IRI(String),
    BlankNode(String),
    Literal {
        value: String,
        datatype: Option<String>,
        lang: Option<String>,
    },
    QuotedTriple {
        subject: Box<EncodedTerm>,
        predicate: Box<EncodedTerm>,
        object: Box<EncodedTerm>,
    },
}

impl EncodedSparqlValue {
    /// Convert back to full SparqlValue for query processing
    pub fn to_sparql_value(&self) -> SparqlValue {
        match self {
            EncodedSparqlValue::IRI(iri) => SparqlValue::iri(iri.clone()),
            EncodedSparqlValue::BlankNode(id) => SparqlValue::blank_node(id.clone()),
            EncodedSparqlValue::Literal {
                value,
                datatype,
                lang,
            } => SparqlValue::literal(value.clone(), datatype.clone(), lang.clone()),
            EncodedSparqlValue::QuotedTriple {
                subject,
                predicate,
                object,
            } => SparqlValue::quoted_triple(
                subject.to_rdf_term(),
                predicate.to_rdf_term(),
                object.to_rdf_term(),
            ),
            EncodedSparqlValue::Unknown(value) => SparqlValue {
                value_type: "unknown".to_string(),
                value: value.clone(),
                datatype: None,
                lang: None,
                quoted_triple: None,
            },
        }
    }
}

impl EncodedTerm {
    /// Convert back to RDF term for processing
    pub fn to_rdf_term(&self) -> RdfTerm {
        match self {
            EncodedTerm::IRI(iri) => RdfTerm::IRI(iri.clone()),
            EncodedTerm::BlankNode(id) => RdfTerm::BlankNode(id.clone()),
            EncodedTerm::Literal {
                value,
                datatype,
                lang,
            } => RdfTerm::Literal {
                value: value.clone(),
                datatype: datatype.clone(),
                lang: lang.clone(),
            },
            EncodedTerm::QuotedTriple {
                subject,
                predicate,
                object,
            } => RdfTerm::QuotedTriple(QuotedTripleValue {
                subject: Box::new(subject.to_rdf_term()),
                predicate: Box::new(predicate.to_rdf_term()),
                object: Box::new(object.to_rdf_term()),
            }),
        }
    }
}

/// GraphQL response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLResponse {
    pub data: serde_json::Value,
    pub errors: Vec<GraphQLError>,
    pub extensions: Option<serde_json::Value>,
}

/// GraphQL error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLError {
    pub message: String,
    pub locations: Option<Vec<GraphQLLocation>>,
    pub path: Option<Vec<serde_json::Value>>,
}

/// GraphQL error location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLLocation {
    pub line: u32,
    pub column: u32,
}

/// Query result data that can hold either SPARQL or GraphQL results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryResultData {
    Sparql(SparqlResults),
    GraphQL(GraphQLResponse),
    ServiceResult(serde_json::Value),
}

/// Runtime statistics for adaptive execution
#[derive(Debug, Clone)]
pub struct RuntimeStatistics {
    pub group_start_times: HashMap<usize, Instant>,
    pub group_durations: HashMap<usize, Duration>,
    pub total_execution_time: Duration,
    pub groups_executed: usize,
    pub total_steps_executed: usize,
    pub successful_steps: usize,
    pub failed_steps: usize,
    pub average_group_time: Duration,
    pub peak_memory_usage: u64,
    pub peak_cpu_usage: f64,
}

impl Default for RuntimeStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeStatistics {
    pub fn new() -> Self {
        Self {
            group_start_times: HashMap::new(),
            group_durations: HashMap::new(),
            total_execution_time: Duration::from_secs(0),
            groups_executed: 0,
            total_steps_executed: 0,
            successful_steps: 0,
            failed_steps: 0,
            average_group_time: Duration::from_secs(0),
            peak_memory_usage: 0,
            peak_cpu_usage: 0.0,
        }
    }

    pub fn update_group_start(&mut self, group_idx: usize, start_time: Instant) {
        self.group_start_times.insert(group_idx, start_time);
    }

    pub fn update_group_end(&mut self, group_idx: usize, duration: Duration) {
        self.group_durations.insert(group_idx, duration);
        self.groups_executed += 1;

        // Update average group time
        let total_duration: Duration = self.group_durations.values().sum();
        self.average_group_time = total_duration / self.groups_executed.max(1) as u32;
    }

    pub fn record_step_result(&mut self, success: bool) {
        self.total_steps_executed += 1;
        if success {
            self.successful_steps += 1;
        } else {
            self.failed_steps += 1;
        }
    }

    pub fn get_success_rate(&self) -> f64 {
        if self.total_steps_executed == 0 {
            0.0
        } else {
            self.successful_steps as f64 / self.total_steps_executed as f64
        }
    }
}

/// Enhanced performance monitor for adaptive execution
#[derive(Debug)]
pub struct EnhancedPerformanceMonitor {
    step_durations: Vec<Duration>,
    parallel_durations: Vec<Duration>,
    sequential_durations: Vec<Duration>,
    memory_usage_samples: Vec<u64>,
    cpu_usage_samples: Vec<f64>,
    network_latency_samples: Vec<Duration>,
    error_counts: HashMap<String, usize>,
    bottlenecks: HashMap<BottleneckType, usize>,
}

impl Default for EnhancedPerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl EnhancedPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            step_durations: Vec::new(),
            parallel_durations: Vec::new(),
            sequential_durations: Vec::new(),
            memory_usage_samples: Vec::new(),
            cpu_usage_samples: Vec::new(),
            network_latency_samples: Vec::new(),
            error_counts: HashMap::new(),
            bottlenecks: HashMap::new(),
        }
    }

    pub fn record_step_execution(&mut self, duration: Duration) {
        self.step_durations.push(duration);
    }

    pub fn record_parallel_execution(&mut self, duration: Duration) {
        self.parallel_durations.push(duration);
    }

    pub fn record_sequential_execution(&mut self, duration: Duration) {
        self.sequential_durations.push(duration);
    }

    pub fn record_memory_usage(&mut self, usage: u64) {
        self.memory_usage_samples.push(usage);
    }

    pub fn record_cpu_usage(&mut self, usage: f64) {
        self.cpu_usage_samples.push(usage);
    }

    pub fn record_network_latency(&mut self, latency: Duration) {
        self.network_latency_samples.push(latency);
    }

    pub fn record_error(&mut self, error_type: String) {
        *self.error_counts.entry(error_type).or_insert(0) += 1;
    }

    pub fn record_bottleneck(&mut self, bottleneck: BottleneckType) {
        *self.bottlenecks.entry(bottleneck).or_insert(0) += 1;
    }

    pub fn get_average_step_time(&self) -> Duration {
        if self.step_durations.is_empty() {
            Duration::from_secs(0)
        } else {
            self.step_durations.iter().sum::<Duration>() / self.step_durations.len() as u32
        }
    }

    pub fn get_average_parallel_time(&self) -> Duration {
        if self.parallel_durations.is_empty() {
            Duration::from_secs(0)
        } else {
            self.parallel_durations.iter().sum::<Duration>() / self.parallel_durations.len() as u32
        }
    }

    pub fn get_average_sequential_time(&self) -> Duration {
        if self.sequential_durations.is_empty() {
            Duration::from_secs(0)
        } else {
            self.sequential_durations.iter().sum::<Duration>()
                / self.sequential_durations.len() as u32
        }
    }

    pub fn get_memory_percentile(&self, percentile: f64) -> u64 {
        if self.memory_usage_samples.is_empty() {
            return 0;
        }

        let mut sorted = self.memory_usage_samples.clone();
        sorted.sort();
        let index = ((sorted.len() as f64 - 1.0) * percentile / 100.0).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }

    pub fn get_cpu_percentile(&self, percentile: f64) -> f64 {
        if self.cpu_usage_samples.is_empty() {
            return 0.0;
        }

        let mut sorted = self.cpu_usage_samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = ((sorted.len() as f64 - 1.0) * percentile / 100.0).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }

    pub fn get_primary_bottleneck(&self) -> Option<BottleneckType> {
        self.bottlenecks
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(bottleneck, _)| *bottleneck)
    }

    pub fn get_error_rate(&self) -> f64 {
        let total_errors: usize = self.error_counts.values().sum();
        let total_executions = self.step_durations.len();

        if total_executions == 0 {
            0.0
        } else {
            total_errors as f64 / total_executions as f64
        }
    }
}

/// Local resource monitor for tracking system resources
#[derive(Debug)]
pub struct LocalResourceMonitor {
    current_memory_usage: u64,
    current_cpu_usage: f64,
    peak_memory_usage: u64,
    peak_cpu_usage: f64,
}

impl Default for LocalResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalResourceMonitor {
    pub fn new() -> Self {
        Self {
            current_memory_usage: 0,
            current_cpu_usage: 0.0,
            peak_memory_usage: 0,
            peak_cpu_usage: 0.0,
        }
    }

    pub fn get_memory_usage(&self) -> u64 {
        // In practice, this would query actual system memory usage
        self.current_memory_usage
    }

    pub fn get_cpu_usage(&self) -> f64 {
        // In practice, this would query actual CPU usage
        self.current_cpu_usage
    }

    pub fn update_memory_usage(&mut self, usage: u64) {
        self.current_memory_usage = usage;
        if usage > self.peak_memory_usage {
            self.peak_memory_usage = usage;
        }
    }

    pub fn update_cpu_usage(&mut self, usage: f64) {
        self.current_cpu_usage = usage;
        if usage > self.peak_cpu_usage {
            self.peak_cpu_usage = usage;
        }
    }
}

/// Local adaptive configuration that evolves during execution
#[derive(Debug, Clone)]
pub struct LocalAdaptiveConfig {
    pub performance_threshold: f64,
    pub memory_threshold: u64,
    pub cpu_threshold: f64,
    pub error_rate_threshold: f64,
    pub reoptimization_interval: usize,
    pub parallel_threshold: usize,
    pub streaming_threshold: usize,
    pub latency_threshold: u128,
    pub hybrid_batch_size: usize,
    pub batch_delay_ms: u64,
}

impl Default for LocalAdaptiveConfig {
    fn default() -> Self {
        Self {
            performance_threshold: 1.5, // 50% performance degradation threshold
            memory_threshold: 1024 * 1024 * 1024, // 1GB memory threshold
            cpu_threshold: 0.8,         // 80% CPU threshold
            error_rate_threshold: 0.1,  // 10% error rate threshold
            reoptimization_interval: 5, // Re-optimize every 5 groups
            parallel_threshold: 3,      // Use parallel execution for 3+ steps
            streaming_threshold: 10,    // Use streaming for 10+ steps
            latency_threshold: 1000,    // 1 second latency threshold
            hybrid_batch_size: 3,       // Process 3 steps per batch in hybrid mode
            batch_delay_ms: 50,         // 50ms delay between batches
        }
    }
}

/// Execution strategies for adaptive execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveExecutionStrategy {
    Parallel,
    Sequential,
    Hybrid,
    Streaming,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BottleneckType {
    NetworkLatency,
    MemoryUsage,
    CpuUsage,
    DiskIo,
}

/// Data about group execution for performance analysis
#[derive(Debug, Clone)]
pub struct GroupExecutionData {
    pub group_size: usize,
    pub duration: Duration,
    pub memory_used: u64,
    pub cpu_used: f64,
}

/// Enhanced step result with monitoring data
#[derive(Debug, Clone)]
pub struct EnhancedStepResult {
    pub step_id: String,
    pub execution_time: Duration,
    pub memory_used: u64,
    pub result_size: usize,
    pub success: bool,
    pub error_message: Option<String>,
    pub service_response_time: Option<Duration>,
    pub cache_hit: bool,
}

/// Configuration for reoptimization decisions
#[derive(Debug, Clone)]
pub struct ReoptimizationConfig {
    pub enable_reoptimization: bool,
    pub reoptimization_threshold: f64,
    pub minimum_group_size: usize,
    pub memory_pressure_threshold: f64,
    pub cpu_pressure_threshold: f64,
    pub error_rate_threshold: f64,
}

impl Default for ReoptimizationConfig {
    fn default() -> Self {
        Self {
            enable_reoptimization: true,
            reoptimization_threshold: 1.5,
            minimum_group_size: 2,
            memory_pressure_threshold: 0.8,
            cpu_pressure_threshold: 0.8,
            error_rate_threshold: 0.1,
        }
    }
}
