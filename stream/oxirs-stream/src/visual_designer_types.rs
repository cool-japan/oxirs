//! # Visual Designer — Types
//!
//! All data types used by the visual pipeline designer: canvas structures,
//! node/edge definitions, pipeline metadata, validation results, and
//! debugger state types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::event::StreamEvent;

/// Visual pipeline designer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualDesignerConfig {
    pub enable_auto_layout: bool,
    pub enable_validation: bool,
    pub enable_optimization: bool,
    pub max_nodes: usize,
    pub max_edges: usize,
    pub enable_real_time_debug: bool,
    pub debug_buffer_size: usize,
    pub enable_profiling: bool,
    pub export_formats: Vec<ExportFormat>,
}

impl Default for VisualDesignerConfig {
    fn default() -> Self {
        Self {
            enable_auto_layout: true,
            enable_validation: true,
            enable_optimization: true,
            max_nodes: 1000,
            max_edges: 5000,
            enable_real_time_debug: true,
            debug_buffer_size: 10000,
            enable_profiling: true,
            export_formats: vec![
                ExportFormat::Json,
                ExportFormat::Yaml,
                ExportFormat::Dot,
                ExportFormat::Mermaid,
            ],
        }
    }
}

/// Supported export formats
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExportFormat {
    Json,
    Yaml,
    Dot,     // GraphViz DOT format
    Mermaid, // Mermaid diagram format
    Svg,     // SVG image
    Png,     // PNG image
}

/// Pipeline node representing a stream operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineNode {
    pub id: String,
    pub name: String,
    pub node_type: NodeType,
    pub position: Position,
    pub config: NodeConfig,
    pub metadata: NodeMetadata,
    pub status: NodeStatus,
}

/// Node types for different stream operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeType {
    // Source nodes
    Source(SourceType),
    // Processing nodes
    Map,
    Filter,
    FlatMap,
    Reduce,
    Aggregate,
    Join,
    Window,
    // Transformation nodes
    Transform(TransformType),
    // ML nodes
    MLModel(MLModelType),
    // Output nodes
    Sink(SinkType),
    // Control nodes
    Router,
    Splitter,
    Merger,
    // Debug nodes
    Breakpoint,
    Logger,
    Profiler,
}

/// Source types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SourceType {
    Kafka,
    Nats,
    Redis,
    Memory,
    File,
    WebSocket,
    Http,
    Custom(String),
}

/// Transform types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TransformType {
    RdfTransform,
    SparqlQuery,
    GraphPattern,
    Custom(String),
}

/// ML model types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MLModelType {
    OnlineLearning,
    AnomalyDetection,
    Prediction,
    Classification,
    Clustering,
    Custom(String),
}

/// Sink types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SinkType {
    Kafka,
    Nats,
    Redis,
    Memory,
    Database,
    File,
    WebSocket,
    Http,
    Custom(String),
}

/// Node position in visual canvas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub x: f64,
    pub y: f64,
    pub z: Option<f64>, // For 3D visualization
}

/// Node configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub parameters: HashMap<String, ConfigValue>,
    pub input_ports: Vec<Port>,
    pub output_ports: Vec<Port>,
    pub resource_limits: ResourceLimits,
}

/// Configuration value types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConfigValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<ConfigValue>),
    Object(HashMap<String, ConfigValue>),
}

/// Port for node connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Port {
    pub id: String,
    pub name: String,
    pub port_type: PortType,
    pub data_type: DataType,
    pub required: bool,
}

/// Port types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PortType {
    Input,
    Output,
}

/// Data types flowing through ports
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DataType {
    StreamEvent,
    RdfTriple,
    SparqlResult,
    Json,
    Binary,
    Custom(String),
}

/// Resource limits for nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_memory_mb: Option<u64>,
    pub max_cpu_percent: Option<f64>,
    pub max_execution_time_ms: Option<u64>,
    pub max_events_per_second: Option<u64>,
}

/// Node metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetadata {
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub version: String,
    pub author: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,
}

/// Node status for monitoring
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeStatus {
    Idle,
    Running,
    Paused,
    Error(String),
    Completed,
}

/// Pipeline edge connecting nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineEdge {
    pub id: String,
    pub source_node_id: String,
    pub source_port_id: String,
    pub target_node_id: String,
    pub target_port_id: String,
    pub edge_type: EdgeType,
    pub config: EdgeConfig,
    pub metadata: EdgeMetadata,
}

/// Edge types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EdgeType {
    DataFlow,
    ControlFlow,
    Conditional(Condition),
}

/// Condition for conditional edges
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Condition {
    pub expression: String,
    pub predicate_type: PredicateType,
}

/// Predicate types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PredicateType {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    Contains,
    Matches,
    Custom(String),
}

/// Edge configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeConfig {
    pub buffer_size: usize,
    pub backpressure_strategy: BackpressureStrategy,
    pub error_handling: ErrorHandling,
}

/// Backpressure strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackpressureStrategy {
    Drop,
    Buffer,
    Block,
    Exponential,
    Adaptive,
}

/// Error handling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorHandling {
    Propagate,
    Ignore,
    Retry { max_attempts: u32 },
    DeadLetter,
    Custom(String),
}

/// Edge metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeMetadata {
    pub created_at: DateTime<Utc>,
    pub label: Option<String>,
    pub style: EdgeStyle,
}

/// Edge visual style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeStyle {
    pub color: String,
    pub thickness: f64,
    pub line_type: LineType,
}

/// Line types for edges
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LineType {
    Solid,
    Dashed,
    Dotted,
    Curved,
}

/// Visual pipeline definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualPipeline {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub version: String,
    pub nodes: HashMap<String, PipelineNode>,
    pub edges: HashMap<String, PipelineEdge>,
    pub metadata: PipelineMetadata,
    pub validation_result: Option<ValidationResult>,
}

/// Pipeline metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetadata {
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub author: Option<String>,
    pub tags: Vec<String>,
    pub properties: HashMap<String, String>,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub validated_at: DateTime<Utc>,
}

/// Validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub error_type: ValidationErrorType,
    pub message: String,
    pub node_id: Option<String>,
    pub edge_id: Option<String>,
}

/// Validation error types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ValidationErrorType {
    MissingRequiredPort,
    IncompatibleDataTypes,
    CyclicDependency,
    DisconnectedNode,
    InvalidConfiguration,
    ResourceLimitExceeded,
    DuplicateNodeId,
}

/// Validation warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    pub warning_type: ValidationWarningType,
    pub message: String,
    pub node_id: Option<String>,
    pub suggestion: Option<String>,
}

/// Validation warning types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ValidationWarningType {
    UnusedPort,
    SuboptimalConfiguration,
    PerformanceBottleneck,
    MemoryPressure,
    DeprecatedNode,
}

/// Pipeline debugger for real-time debugging
#[derive(Debug)]
pub struct PipelineDebugger {
    pub pipeline: VisualPipeline,
    pub config: DebuggerConfig,
    pub state: Arc<RwLock<DebuggerState>>,
    pub breakpoints: Arc<RwLock<HashMap<String, Breakpoint>>>,
    pub event_history: Arc<RwLock<VecDeque<DebugEvent>>>,
}

/// Debugger configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebuggerConfig {
    pub enable_breakpoints: bool,
    pub enable_event_capture: bool,
    pub max_event_history: usize,
    pub enable_time_travel: bool,
    pub enable_profiling: bool,
    pub capture_intermediate_results: bool,
}

impl Default for DebuggerConfig {
    fn default() -> Self {
        Self {
            enable_breakpoints: true,
            enable_event_capture: true,
            max_event_history: 10000,
            enable_time_travel: true,
            enable_profiling: true,
            capture_intermediate_results: true,
        }
    }
}

/// Debugger state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebuggerState {
    pub is_running: bool,
    pub is_paused: bool,
    pub current_node_id: Option<String>,
    pub execution_stack: Vec<String>,
    pub variables: HashMap<String, DebugVariable>,
    pub metrics: DebugMetrics,
}

/// Debug variable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugVariable {
    pub name: String,
    pub value: String,
    pub var_type: String,
    pub scope: String,
}

/// Debug metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DebugMetrics {
    pub events_processed: u64,
    pub events_dropped: u64,
    pub average_latency_ms: f64,
    pub throughput_per_second: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
}

/// Breakpoint for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Breakpoint {
    pub id: String,
    pub node_id: String,
    pub condition: Option<String>,
    pub enabled: bool,
    pub hit_count: u64,
    pub max_hits: Option<u64>,
}

/// Debug event captured during execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugEvent {
    pub timestamp: DateTime<Utc>,
    pub node_id: String,
    pub event_type: DebugEventType,
    pub data: StreamEvent,
    pub metadata: HashMap<String, String>,
}

/// Debug event types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DebugEventType {
    NodeEnter,
    NodeExit,
    BreakpointHit,
    Error,
    Warning,
}

/// Pipeline information summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub node_count: usize,
    pub edge_count: usize,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub original_metrics: PipelineMetrics,
    pub suggestions: Vec<OptimizationSuggestion>,
    pub optimized_at: DateTime<Utc>,
}

/// Pipeline metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetrics {
    pub node_count: usize,
    pub edge_count: usize,
    pub avg_chain_length: f64,
    pub max_chain_length: usize,
    pub parallel_opportunities: usize,
    pub bottleneck_nodes: Vec<String>,
}

/// Optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    pub suggestion_type: OptimizationType,
    pub impact: ImpactLevel,
    pub description: String,
    pub estimated_improvement: f64,
}

/// Optimization types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OptimizationType {
    ReduceChainLength,
    IncreaseParallelism,
    OptimizeBufferSize,
    ReduceMemoryUsage,
    ImproveLocality,
}

/// Impact levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImpactLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Pipeline validator
pub struct PipelineValidator {
    pub(crate) config: VisualDesignerConfig,
}

/// Pipeline optimizer
pub struct PipelineOptimizer {
    pub(crate) config: VisualDesignerConfig,
}
