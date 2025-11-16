//! Core types for OxiRS Chat
//!
//! This module contains the fundamental types used throughout the chat system,
//! including messages, rich content elements, and associated metadata.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Chat session configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatConfig {
    pub max_context_length: usize,
    pub temperature: f32,
    pub max_retrieval_results: usize,
    pub enable_sparql_generation: bool,
    pub session_timeout: Duration,
    pub max_conversation_turns: usize,
    pub enable_context_summarization: bool,
    pub sliding_window_size: usize,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            max_context_length: 4096,
            temperature: 0.7,
            max_retrieval_results: 10,
            enable_sparql_generation: true,
            session_timeout: Duration::from_secs(3600),
            max_conversation_turns: 100,
            enable_context_summarization: true,
            sliding_window_size: 20,
        }
    }
}

/// Chat message with rich content support
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub id: String,
    pub role: MessageRole,
    pub content: MessageContent,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: Option<MessageMetadata>,
    pub thread_id: Option<String>,
    pub parent_message_id: Option<String>,
    pub token_count: Option<usize>,
    pub reactions: Vec<MessageReaction>,
    pub attachments: Vec<MessageAttachment>,
    pub rich_elements: Vec<RichContentElement>,
}

/// Message content supporting both plain text and rich content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageContent {
    /// Plain text content
    Text(String),
    /// Rich content with multiple elements
    Rich {
        text: String,
        elements: Vec<RichContentElement>,
    },
}

impl MessageContent {
    pub fn to_text(&self) -> &str {
        match self {
            MessageContent::Text(text) => text,
            MessageContent::Rich { text, .. } => text,
        }
    }

    pub fn from_text(text: String) -> Self {
        MessageContent::Text(text)
    }

    pub fn add_element(&mut self, element: RichContentElement) {
        match self {
            MessageContent::Text(text) => {
                let text = std::mem::take(text);
                *self = MessageContent::Rich {
                    text,
                    elements: vec![element],
                };
            }
            MessageContent::Rich { elements, .. } => {
                elements.push(element);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.to_text().len()
    }

    pub fn contains(&self, pat: char) -> bool {
        self.to_text().contains(pat)
    }

    pub fn to_lowercase(&self) -> String {
        self.to_text().to_lowercase()
    }

    pub fn chars(&self) -> std::str::Chars<'_> {
        self.to_text().chars()
    }

    pub fn is_empty(&self) -> bool {
        self.to_text().is_empty()
    }
}

impl std::fmt::Display for MessageContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

/// Message role enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

/// Message metadata for analytics and context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMetadata {
    pub session_id: String,
    pub turn_number: usize,
    pub processing_time_ms: Option<u64>,
    pub retrieval_results: Option<usize>,
    pub sparql_query: Option<String>,
    pub confidence_score: Option<f32>,
    pub intent: Option<String>,
    pub entities: Vec<String>,
    pub topics: Vec<String>,
    pub quality_score: Option<f32>,
    pub user_feedback: Option<UserFeedback>,
    pub error_details: Option<ErrorDetails>,
}

/// User feedback for quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFeedback {
    pub rating: u8, // 1-5 stars
    pub helpful: Option<bool>,
    pub comment: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Error details for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetails {
    pub error_type: String,
    pub error_message: String,
    pub error_code: Option<String>,
    pub stack_trace: Option<String>,
    pub recovery_suggestions: Vec<String>,
}

/// Message reaction for user engagement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageReaction {
    pub reaction_type: ReactionType,
    pub user_id: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Reaction type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReactionType {
    Like,
    Dislike,
    Helpful,
    NotHelpful,
    Accurate,
    Inaccurate,
    Clear,
    Confusing,
}

/// Message attachment for file uploads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageAttachment {
    pub id: String,
    pub filename: String,
    pub file_type: String,
    pub size_bytes: u64,
    pub url: Option<String>,
    pub thumbnail_url: Option<String>,
    pub metadata: AttachmentMetadata,
    pub upload_timestamp: chrono::DateTime<chrono::Utc>,
    pub processing_status: AttachmentProcessingStatus,
}

/// Attachment metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachmentMetadata {
    pub extracted_text: Option<String>,
    pub language: Option<String>,
    pub format_detected: Option<String>,
    pub processing_notes: Vec<String>,
}

/// Attachment processing status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttachmentProcessingStatus {
    Pending,
    Processing,
    Completed,
    Failed(String),
    PartiallyProcessed(String),
}

/// Rich content elements that can be embedded in messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RichContentElement {
    /// Code snippet with syntax highlighting
    CodeBlock {
        language: String,
        code: String,
        title: Option<String>,
        line_numbers: bool,
        highlight_lines: Vec<usize>,
    },
    /// SPARQL query block with execution metadata
    SparqlQuery {
        query: String,
        execution_time_ms: Option<u64>,
        result_count: Option<usize>,
        status: QueryExecutionStatus,
        explanation: Option<String>,
    },
    /// Data table with formatting options
    Table {
        headers: Vec<String>,
        rows: Vec<Vec<String>>,
        title: Option<String>,
        pagination: Option<TablePagination>,
        sorting: Option<TableSorting>,
        formatting: TableFormatting,
    },
    /// Graph visualization configuration
    GraphVisualization {
        graph_type: GraphType,
        data: GraphData,
        layout: GraphLayout,
        styling: GraphStyling,
        interactive: bool,
    },
    /// Chart or plot
    Chart {
        chart_type: ChartType,
        data: ChartData,
        title: Option<String>,
        axes: ChartAxes,
        styling: ChartStyling,
    },
    /// File upload reference
    FileReference {
        file_id: String,
        filename: String,
        file_type: String,
        size_bytes: u64,
        preview: Option<FilePreview>,
    },
    /// Interactive widget
    Widget {
        widget_type: WidgetType,
        data: serde_json::Value,
        config: WidgetConfig,
    },
    /// Timeline visualization
    Timeline {
        events: Vec<TimelineEvent>,
        range: TimelineRange,
        styling: TimelineStyling,
    },
    /// Quantum-enhanced search results visualization
    QuantumVisualization {
        results: Vec<QuantumSearchResult>,
        entanglement_map: std::collections::HashMap<String, f32>,
    },
    /// Consciousness-aware processing insights
    ConsciousnessInsights {
        insights: Vec<ConsciousnessInsight>,
        awareness_level: f32,
    },
    /// Advanced reasoning chain visualization
    ReasoningChain {
        reasoning_steps: Vec<ReasoningStep>,
        confidence_score: f32,
    },
    /// SPARQL query results with execution metadata
    SPARQLResults {
        query: String,
        results: Vec<std::collections::HashMap<String, String>>,
        execution_time: Duration,
    },
}

/// Query execution status for SPARQL queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryExecutionStatus {
    Success,
    Error(String),
    Timeout,
    Cancelled,
    ValidationError(String),
}

/// Table pagination information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TablePagination {
    pub current_page: usize,
    pub total_pages: usize,
    pub page_size: usize,
    pub total_rows: usize,
}

/// Table sorting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSorting {
    pub column: String,
    pub direction: SortDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// Table formatting options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableFormatting {
    pub striped_rows: bool,
    pub borders: bool,
    pub compact: bool,
    pub theme: TableTheme,
    pub cell_padding: CellPadding,
    pub column_widths: Vec<ColumnWidth>,
}

impl Default for TableFormatting {
    fn default() -> Self {
        Self {
            striped_rows: true,
            borders: true,
            compact: false,
            theme: TableTheme::Default,
            cell_padding: CellPadding::Medium,
            column_widths: vec![],
        }
    }
}

/// Table theme enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TableTheme {
    Default,
    Dark,
    Light,
    Minimal,
    Professional,
}

/// Cell padding options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CellPadding {
    None,
    Small,
    Medium,
    Large,
}

/// Column width specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColumnWidth {
    Auto,
    Fixed(u32),
    Percentage(f32),
    MinMax { min: u32, max: u32 },
}

/// Graph type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphType {
    Network,
    Tree,
    Hierarchy,
    Flow,
    Timeline,
    Circular,
    Force,
}

/// Graph data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphData {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub metadata: GraphMetadata,
}

/// Graph node representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub node_type: String,
    pub properties: std::collections::HashMap<String, serde_json::Value>,
    pub styling: NodeStyling,
}

/// Graph edge representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub label: Option<String>,
    pub edge_type: String,
    pub weight: Option<f64>,
    pub properties: std::collections::HashMap<String, serde_json::Value>,
    pub styling: EdgeStyling,
}

/// Graph metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    pub title: Option<String>,
    pub description: Option<String>,
    pub source: Option<String>,
    pub node_count: usize,
    pub edge_count: usize,
    pub creation_time: chrono::DateTime<chrono::Utc>,
}

/// Graph layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphLayout {
    pub algorithm: LayoutAlgorithm,
    pub parameters: std::collections::HashMap<String, serde_json::Value>,
    pub constraints: Vec<LayoutConstraint>,
}

/// Layout algorithm enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayoutAlgorithm {
    ForceDirected,
    Hierarchical,
    Circular,
    Grid,
    Random,
    Manual,
}

/// Layout constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutConstraint {
    pub constraint_type: ConstraintType,
    pub target: String,
    pub parameters: std::collections::HashMap<String, serde_json::Value>,
}

/// Constraint type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    FixedPosition,
    MinDistance,
    MaxDistance,
    Alignment,
    Grouping,
}

/// Graph styling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStyling {
    pub theme: GraphTheme,
    pub colors: ColorScheme,
    pub fonts: FontConfiguration,
    pub effects: VisualEffects,
}

/// Graph theme enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphTheme {
    Light,
    Dark,
    HighContrast,
    Minimal,
    Professional,
}

/// Color scheme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorScheme {
    pub primary: String,
    pub secondary: String,
    pub accent: String,
    pub background: String,
    pub text: String,
    pub edge: String,
}

/// Font configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontConfiguration {
    pub family: String,
    pub size: u32,
    pub weight: FontWeight,
    pub style: FontStyle,
}

/// Font weight enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontWeight {
    Normal,
    Bold,
    Light,
    ExtraBold,
}

/// Font style enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontStyle {
    Normal,
    Italic,
    Oblique,
}

/// Visual effects configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualEffects {
    pub shadows: bool,
    pub animations: bool,
    pub hover_effects: bool,
    pub selection_highlight: bool,
    pub zoom_effects: bool,
}

/// Node styling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStyling {
    pub shape: NodeShape,
    pub size: NodeSize,
    pub color: String,
    pub border: BorderStyle,
    pub icon: Option<String>,
}

/// Node shape enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeShape {
    Circle,
    Square,
    Triangle,
    Diamond,
    Hexagon,
    Star,
}

/// Node size specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeSize {
    Small,
    Medium,
    Large,
    Custom(u32),
}

/// Border style configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorderStyle {
    pub width: u32,
    pub color: String,
    pub style: LineStyle,
}

/// Line style enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LineStyle {
    Solid,
    Dashed,
    Dotted,
    Double,
}

/// Edge styling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeStyling {
    pub line_style: LineStyle,
    pub color: String,
    pub width: u32,
    pub arrow: ArrowStyle,
    pub curvature: Curvature,
}

/// Arrow style enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArrowStyle {
    None,
    Simple,
    Filled,
    Open,
    Diamond,
}

/// Curvature specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Curvature {
    Straight,
    Curved(f32),
    Bezier { cp1: (f32, f32), cp2: (f32, f32) },
}

/// Chart type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartType {
    Line,
    Bar,
    Pie,
    Scatter,
    Area,
    Histogram,
    Box,
    Violin,
    Heatmap,
}

/// Chart data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartData {
    pub series: Vec<DataSeries>,
    pub categories: Vec<String>,
    pub metadata: ChartMetadata,
}

/// Data series for charts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSeries {
    pub name: String,
    pub data: Vec<DataPoint>,
    pub color: Option<String>,
    pub line_style: Option<LineStyle>,
}

/// Data point representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub x: serde_json::Value,
    pub y: serde_json::Value,
    pub label: Option<String>,
    pub metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
}

/// Chart metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartMetadata {
    pub source: Option<String>,
    pub description: Option<String>,
    pub units: Option<String>,
    pub creation_time: chrono::DateTime<chrono::Utc>,
}

/// Chart axes configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartAxes {
    pub x_axis: AxisConfiguration,
    pub y_axis: AxisConfiguration,
    pub secondary_y: Option<AxisConfiguration>,
}

/// Axis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisConfiguration {
    pub title: Option<String>,
    pub scale: AxisScale,
    pub range: Option<AxisRange>,
    pub tick_format: Option<String>,
    pub grid_lines: bool,
}

/// Axis scale enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AxisScale {
    Linear,
    Logarithmic,
    Time,
    Category,
}

/// Axis range specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisRange {
    pub min: serde_json::Value,
    pub max: serde_json::Value,
}

/// Chart styling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartStyling {
    pub theme: ChartTheme,
    pub colors: Vec<String>,
    pub fonts: FontConfiguration,
    pub legend: LegendConfiguration,
}

/// Chart theme enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartTheme {
    Default,
    Dark,
    Light,
    Minimal,
    Professional,
}

/// Legend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegendConfiguration {
    pub position: LegendPosition,
    pub visible: bool,
    pub orientation: LegendOrientation,
}

/// Legend position enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LegendPosition {
    Top,
    Bottom,
    Left,
    Right,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// Legend orientation enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LegendOrientation {
    Horizontal,
    Vertical,
}

/// File preview for attachments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilePreview {
    pub preview_type: PreviewType,
    pub content: String,
    pub thumbnail_url: Option<String>,
}

/// Preview type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreviewType {
    Text,
    Image,
    Video,
    Audio,
    Document,
    Code,
}

/// Widget type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetType {
    Slider,
    Button,
    Input,
    Dropdown,
    Toggle,
    DatePicker,
    ColorPicker,
    Progress,
}

/// Widget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetConfig {
    pub interactive: bool,
    pub validation: Option<ValidationRules>,
    pub events: Vec<WidgetEvent>,
}

/// Validation rules for widgets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRules {
    pub required: bool,
    pub min_value: Option<serde_json::Value>,
    pub max_value: Option<serde_json::Value>,
    pub pattern: Option<String>,
    pub custom_validator: Option<String>,
}

/// Widget event configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetEvent {
    pub event_type: String,
    pub handler: String,
    pub parameters: std::collections::HashMap<String, serde_json::Value>,
}

/// Timeline event representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub duration: Option<Duration>,
    pub event_type: String,
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

/// Timeline range specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineRange {
    pub start: chrono::DateTime<chrono::Utc>,
    pub end: chrono::DateTime<chrono::Utc>,
    pub scale: TimelineScale,
}

/// Timeline scale enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimelineScale {
    Minutes,
    Hours,
    Days,
    Weeks,
    Months,
    Years,
}

/// Timeline styling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineStyling {
    pub theme: TimelineTheme,
    pub orientation: TimelineOrientation,
    pub marker_style: MarkerStyle,
    pub line_style: LineStyle,
}

/// Timeline theme enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimelineTheme {
    Default,
    Minimal,
    Detailed,
    Compact,
}

/// Timeline orientation enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimelineOrientation {
    Horizontal,
    Vertical,
}

/// Marker style for timeline events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkerStyle {
    pub shape: MarkerShape,
    pub size: u32,
    pub color: String,
    pub border: BorderStyle,
}

/// Marker shape enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarkerShape {
    Circle,
    Square,
    Triangle,
    Diamond,
    Star,
}

/// Streaming response chunk for real-time communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamResponseChunk {
    /// Status update with processing stage and progress
    Status {
        stage: ProcessingStage,
        progress: f32, // 0.0 to 1.0
        message: Option<String>,
    },
    /// Context information found during retrieval
    Context {
        facts: Vec<String>,
        sparql_results: Option<std::collections::HashMap<String, String>>,
        entities: Vec<String>,
    },
    /// Incremental content being generated
    Content { text: String, is_complete: bool },
    /// Error occurred during processing
    Error {
        error: StructuredError,
        recoverable: bool,
    },
    /// Processing complete with final message
    Complete {
        total_time: Duration,
        token_count: usize,
        final_message: Option<String>,
    },
}

/// Stream processing stage identifiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStage {
    Initializing,
    RetrievingContext,
    GeneratingSparql,
    ExecutingQuery,
    QuantumProcessing,
    ConsciousnessProcessing,
    AdvancedReasoning,
    GeneratingResponse,
    Finalizing,
}

impl ProcessingStage {
    /// Get the human-readable name for the stage
    pub fn display_name(&self) -> &'static str {
        match self {
            ProcessingStage::Initializing => "Initializing",
            ProcessingStage::RetrievingContext => "Retrieving Context",
            ProcessingStage::GeneratingSparql => "Generating SPARQL",
            ProcessingStage::ExecutingQuery => "Executing Query",
            ProcessingStage::QuantumProcessing => "Quantum Processing",
            ProcessingStage::ConsciousnessProcessing => "Consciousness Processing",
            ProcessingStage::AdvancedReasoning => "Advanced Reasoning",
            ProcessingStage::GeneratingResponse => "Generating Response",
            ProcessingStage::Finalizing => "Finalizing",
        }
    }

    /// Get expected progress for each stage
    pub fn expected_progress(&self) -> f32 {
        match self {
            ProcessingStage::Initializing => 0.0,
            ProcessingStage::RetrievingContext => 0.1,
            ProcessingStage::GeneratingSparql => 0.3,
            ProcessingStage::ExecutingQuery => 0.5,
            ProcessingStage::QuantumProcessing => 0.6,
            ProcessingStage::ConsciousnessProcessing => 0.7,
            ProcessingStage::AdvancedReasoning => 0.8,
            ProcessingStage::GeneratingResponse => 0.9,
            ProcessingStage::Finalizing => 1.0,
        }
    }
}

/// Quantum search result for quantum-enhanced visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSearchResult {
    pub triple: String,
    pub score: f32,
    pub quantum_amplitude: f32,
    pub phase: f32,
    pub entanglement_factor: f32,
    pub coherence_time: f32,
}

/// Consciousness processing insight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessInsight {
    pub content: String,
    pub confidence: f32,
    pub insight_type: ConsciousnessInsightType,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Type of consciousness insight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsciousnessInsightType {
    MemoryTrace,
    EmotionalResonance,
    AttentionFocus,
    MetacognitiveLearning,
    TemporalCoherence,
}

/// Reasoning step in an advanced reasoning chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub reasoning_type: ReasoningType,
    pub premise_triples: Vec<String>,
    pub conclusion_triple: Option<String>,
    pub confidence: f32,
    pub explanation: String,
}

/// Type of reasoning used in a step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningType {
    Deductive,
    Inductive,
    Causal,
    Temporal,
    Analogical,
    Probabilistic,
}

/// Enhanced streaming response chunk with more detailed error handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnhancedStreamResponseChunk {
    /// Status update with detailed stage information
    Status {
        stage: ProcessingStage,
        progress: f32,
        stage_details: String,
        estimated_remaining_ms: Option<u64>,
    },
    /// Context information with categorization
    Context {
        content: String,
        context_type: ContextType,
        confidence: f32,
    },
    /// Incremental content with metadata
    Content {
        content: String,
        is_final: bool,
        content_type: ContentType,
        word_count: usize,
    },
    /// Structured error with recovery suggestions
    Error {
        error: StructuredError,
        recovery_suggestions: Vec<String>,
        can_retry: bool,
    },
    /// Processing complete with comprehensive metadata
    Complete {
        message: Box<crate::Message>,
        total_time: Duration,
        performance_metrics: ProcessingMetrics,
    },
}

/// Type of context information being streamed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextType {
    KnowledgeGraphFacts,
    EntityExtraction,
    SparqlGeneration,
    SparqlExecution,
    QuantumEnhancement,
    ConsciousnessInsights,
    ReasoningAnalysis,
}

/// Type of content being streamed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentType {
    PlainText,
    FormattedText,
    RichContent,
    CodeBlock,
    Table,
    Visualization,
}

/// Structured error information for better error handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredError {
    pub error_type: ErrorType,
    pub message: String,
    pub error_code: Option<String>,
    pub component: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub context: std::collections::HashMap<String, serde_json::Value>,
}

/// Type of error that occurred
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorType {
    RagRetrievalError,
    SparqlGenerationError,
    SparqlExecutionError,
    LlmGenerationError,
    QuantumProcessingError,
    ConsciousnessProcessingError,
    ReasoningError,
    NetworkError,
    TimeoutError,
    ValidationError,
    InternalError,
}

/// Performance metrics for processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetrics {
    pub total_time_ms: u64,
    pub rag_retrieval_time_ms: u64,
    pub sparql_generation_time_ms: u64,
    pub sparql_execution_time_ms: u64,
    pub llm_generation_time_ms: u64,
    pub quantum_processing_time_ms: u64,
    pub consciousness_processing_time_ms: u64,
    pub reasoning_time_ms: u64,
    pub memory_usage_mb: f32,
    pub cpu_usage_percent: f32,
    pub cache_hit_rate: f32,
    pub token_count: usize,
    pub request_size_bytes: usize,
    pub response_size_bytes: usize,
}

// Enhanced context preservation types

/// Conversation analysis for intelligent context compression
#[derive(Debug, Default, Clone)]
pub struct ConversationAnalysis {
    pub question_count: usize,
    pub resolved_questions: usize,
    pub unresolved_questions: usize,
    pub assistant_response_count: usize,
    pub error_mentions: usize,
    pub code_examples: usize,
}

/// Key concept extracted from conversation
#[derive(Debug, Clone)]
pub struct KeyConcept {
    pub name: String,
    pub frequency: usize,
    pub importance: f32,
    pub context: String,
}

/// Key outcome or decision from conversation
#[derive(Debug, Clone)]
pub struct KeyOutcome {
    pub description: String,
    pub message_id: String,
    pub outcome_type: OutcomeType,
    pub confidence: f32,
}

/// Type of outcome identified in conversation
#[derive(Debug, Clone)]
pub enum OutcomeType {
    Solution,
    Decision,
    Example,
    Error,
    Recommendation,
}

/// User interaction patterns for personalization
#[derive(Debug, Default, Clone)]
pub struct InteractionPatterns {
    pub user_message_count: usize,
    pub average_user_message_length: usize,
    pub complex_questions: usize,
    pub simple_questions: usize,
    pub technical_messages: usize,
    pub preferred_response_style: Option<String>,
}

// Additional types for compatibility with server.rs

/// Thread information for message threads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadInfo {
    pub thread_id: String,
    pub title: Option<String>,
    pub message_count: usize,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
}

/// Session statistics for system monitoring
// Updated SessionStats with all fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStats {
    pub total_sessions: usize,
    pub active_sessions: usize,
    pub idle_sessions: usize,
    pub expired_sessions: usize,
    pub suspended_sessions: usize,
    pub total_messages: usize,
    pub total_tokens: usize,
    pub avg_response_time_ms: f64,
    pub uptime_seconds: u64,
}
