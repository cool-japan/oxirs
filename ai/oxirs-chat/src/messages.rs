//! Message types and rich content support for chat interface

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    pub reactions: Vec<crate::types::MessageReaction>,
    pub attachments: Vec<MessageAttachment>,
    pub rich_elements: Vec<RichContentElement>,
}

/// Message role enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MessageRole {
    User,
    Assistant,
    System,
    Function,
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
    /// Advanced quantum-enhanced search results visualization
    QuantumVisualization {
        results: Vec<crate::rag::quantum_rag::QuantumSearchResult>,
        entanglement_map: HashMap<String, f64>,
    },
    /// Consciousness-aware insights from advanced AI processing
    ConsciousnessInsights {
        insights: Vec<crate::rag::consciousness::ConsciousInsight>,
        awareness_level: f64,
    },
    /// Advanced reasoning chain visualization
    ReasoningChain {
        reasoning_steps: Vec<crate::rag::advanced_reasoning::ReasoningStep>,
        confidence_score: f64,
    },
    /// SPARQL query results with execution details
    SPARQLResults {
        query: String,
        results: Vec<HashMap<String, String>>,
        execution_time: std::time::Duration,
    },
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
    pub highlight_row: Option<usize>,
    pub column_widths: Option<Vec<String>>,
}

impl Default for TableFormatting {
    fn default() -> Self {
        Self {
            striped_rows: true,
            borders: true,
            compact: false,
            highlight_row: None,
            column_widths: None,
        }
    }
}

/// Message metadata for additional information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMetadata {
    pub source: Option<String>,
    pub confidence: Option<f64>,
    pub processing_time_ms: Option<u64>,
    pub model_used: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub custom_fields: HashMap<String, serde_json::Value>,
}

// MessageReaction is now defined in types.rs and re-exported from crate root

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReactionType {
    Like,
    Dislike,
    Helpful,
    NotHelpful,
    Accurate,
    Inaccurate,
    Custom(String),
}

/// Attachment metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachmentMetadata {
    pub extracted_text: Option<String>,
    pub language: Option<String>,
    pub encoding: Option<String>,
    pub checksum: Option<String>,
    pub analysis_results: HashMap<String, serde_json::Value>,
}

/// Attachment processing status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttachmentProcessingStatus {
    Pending,
    Processing,
    Complete,
    Failed(String),
    VirusScanFailed,
    Quarantined,
}

// Graph and Chart related types (simplified for now)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphType {
    KnowledgeGraph,
    NetworkGraph,
    TreeGraph,
    FlowChart,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphData {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub node_type: String,
    pub properties: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub label: Option<String>,
    pub edge_type: String,
    pub properties: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphLayout {
    pub algorithm: String,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStyling {
    pub node_colors: HashMap<String, String>,
    pub edge_colors: HashMap<String, String>,
    pub theme: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartType {
    Line,
    Bar,
    Pie,
    Scatter,
    Histogram,
    Heatmap,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartData {
    pub datasets: Vec<ChartDataset>,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartDataset {
    pub label: String,
    pub data: Vec<f64>,
    pub color: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartAxes {
    pub x_axis: AxisConfig,
    pub y_axis: AxisConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisConfig {
    pub label: String,
    pub scale: AxisScale,
    pub range: Option<(f64, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AxisScale {
    Linear,
    Logarithmic,
    Time,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartStyling {
    pub theme: String,
    pub colors: Vec<String>,
    pub font_family: String,
    pub font_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilePreview {
    pub preview_type: PreviewType,
    pub content: String,
    pub thumbnail_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreviewType {
    Text,
    Image,
    Audio,
    Video,
    Document,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetType {
    Button,
    Slider,
    TextInput,
    Dropdown,
    Checkbox,
    DatePicker,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetConfig {
    pub interactive: bool,
    pub callback_url: Option<String>,
    pub validation: Option<ValidationConfig>,
    pub styling: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub required: bool,
    pub pattern: Option<String>,
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub duration: Option<chrono::Duration>,
    pub event_type: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineRange {
    pub start: chrono::DateTime<chrono::Utc>,
    pub end: chrono::DateTime<chrono::Utc>,
    pub zoom_level: TimelineZoom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimelineZoom {
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Year,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineStyling {
    pub theme: String,
    pub event_colors: HashMap<String, String>,
    pub show_grid: bool,
    pub compact_view: bool,
}
