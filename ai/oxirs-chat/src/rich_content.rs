//! Rich content support for oxirs-chat
//!
//! This module provides support for rich content types including:
//! - Code snippets with syntax highlighting
//! - SPARQL query blocks with validation
//! - Graph visualizations
//! - Table outputs
//! - Image attachments
//! - File uploads

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Chat-specific error type
#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    #[error("Processing error: {0}")]
    Processing(String),
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Chat result type alias
pub type ChatResult<T> = Result<T, ChatError>;

/// Rich content types supported in chat messages
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RichContent {
    /// Plain text content
    Text(String),
    /// Code snippet with syntax highlighting
    CodeSnippet {
        code: String,
        language: String,
        filename: Option<String>,
        line_numbers: bool,
        highlight_lines: Vec<usize>,
        theme: CodeTheme,
    },
    /// SPARQL query block with validation
    SparqlQuery {
        query: String,
        valid: bool,
        error_message: Option<String>,
        execution_plan: Option<String>,
        performance_stats: Option<QueryPerformanceStats>,
    },
    /// Advanced graph visualization data
    GraphVisualization {
        nodes: Vec<GraphNode>,
        edges: Vec<GraphEdge>,
        layout: GraphLayout,
        styling: GraphStyling,
        interactive_features: InteractiveFeatures,
        metadata: HashMap<String, String>,
    },
    /// Enhanced table output with advanced features
    Table {
        headers: Vec<TableHeader>,
        rows: Vec<TableRow>,
        pagination: Option<TablePagination>,
        sorting: Option<TableSorting>,
        filtering: Option<TableFiltering>,
        styling: TableStyling,
        metadata: HashMap<String, String>,
    },
    /// Interactive chart visualization
    Chart {
        chart_type: ChartType,
        data: ChartData,
        configuration: ChartConfiguration,
        styling: ChartStyling,
        interactive_features: ChartInteractivity,
    },
    /// Timeline visualization
    Timeline {
        events: Vec<TimelineEvent>,
        range: TimelineRange,
        styling: TimelineStyling,
        zoom_levels: Vec<TimelineZoomLevel>,
        interactive: bool,
    },
    /// Geographic map visualization
    Map {
        map_type: MapType,
        center: GeoCoordinate,
        zoom_level: u8,
        markers: Vec<MapMarker>,
        layers: Vec<MapLayer>,
        styling: MapStyling,
    },
    /// 3D visualization
    ThreeDVisualization {
        objects: Vec<ThreeDObject>,
        camera: CameraSettings,
        lighting: LightingSettings,
        materials: Vec<Material>,
        animations: Vec<Animation>,
    },
    /// Data dashboard with multiple widgets
    Dashboard {
        widgets: Vec<DashboardWidget>,
        layout: DashboardLayout,
        refresh_interval: Option<std::time::Duration>,
        real_time_updates: bool,
    },
    /// Image attachment with annotations
    Image {
        url: String,
        alt_text: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        annotations: Vec<ImageAnnotation>,
        filters: Vec<ImageFilter>,
    },
    /// Audio content
    Audio {
        url: String,
        duration: Option<std::time::Duration>,
        waveform: Option<Vec<f32>>,
        transcript: Option<String>,
    },
    /// Video content
    Video {
        url: String,
        duration: Option<std::time::Duration>,
        thumbnail: Option<String>,
        subtitles: Vec<SubtitleTrack>,
        chapters: Vec<VideoChapter>,
    },
    /// File upload with preview
    File {
        path: PathBuf,
        filename: String,
        mime_type: String,
        size: u64,
        preview: Option<FilePreview>,
        metadata: FileMetadata,
    },
    /// Interactive widget
    Widget {
        widget_type: WidgetType,
        configuration: serde_json::Value,
        state: serde_json::Value,
        interactive: bool,
    },
}

// Code visualization types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CodeTheme {
    Light,
    Dark,
    HighContrast,
    Monokai,
    Solarized,
    GitHub,
    VSCode,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryPerformanceStats {
    pub execution_time_ms: u64,
    pub result_count: usize,
    pub memory_used_mb: f64,
    pub query_plan_complexity: u32,
}

// Graph visualization types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub node_type: String,
    pub properties: HashMap<String, String>,
    pub position: Option<NodePosition>,
    pub styling: NodeStyling,
    pub size: f64,
    pub shape: NodeShape,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub label: String,
    pub edge_type: String,
    pub properties: HashMap<String, String>,
    pub styling: EdgeStyling,
    pub weight: f64,
    pub directed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodePosition {
    pub x: f64,
    pub y: f64,
    pub z: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeStyling {
    pub color: String,
    pub border_color: String,
    pub border_width: f64,
    pub opacity: f64,
    pub font_size: f64,
    pub font_color: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeShape {
    Circle,
    Square,
    Triangle,
    Diamond,
    Star,
    Hexagon,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeStyling {
    pub color: String,
    pub width: f64,
    pub opacity: f64,
    pub style: EdgeStyle,
    pub arrow_type: ArrowType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EdgeStyle {
    Solid,
    Dashed,
    Dotted,
    Curved,
    Straight,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ArrowType {
    None,
    Standard,
    Filled,
    Open,
    Diamond,
    Circle,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GraphLayout {
    ForceDirected,
    Circular,
    Hierarchical,
    Grid,
    Tree,
    Cluster,
    Custom {
        algorithm: String,
        parameters: HashMap<String, f64>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphStyling {
    pub background_color: String,
    pub grid_enabled: bool,
    pub physics_enabled: bool,
    pub clustering_enabled: bool,
    pub smooth_curves: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InteractiveFeatures {
    pub pan_enabled: bool,
    pub zoom_enabled: bool,
    pub node_selection: bool,
    pub edge_selection: bool,
    pub hover_effects: bool,
    pub click_to_expand: bool,
    pub context_menu: bool,
}

/// Rich content message that can contain multiple content blocks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RichMessage {
    pub content_blocks: Vec<RichContent>,
    pub metadata: HashMap<String, String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl RichMessage {
    /// Create a new rich message
    pub fn new() -> Self {
        Self {
            content_blocks: Vec::new(),
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Add a content block to the message
    pub fn add_content(&mut self, content: RichContent) {
        self.content_blocks.push(content);
    }

    /// Add a text block
    pub fn add_text(&mut self, text: impl Into<String>) {
        self.add_content(RichContent::Text(text.into()));
    }

    /// Add a code snippet
    pub fn add_code(&mut self, code: impl Into<String>, language: impl Into<String>) {
        self.add_content(RichContent::CodeSnippet {
            code: code.into(),
            language: language.into(),
            filename: None,
        });
    }

    /// Add a SPARQL query block
    pub fn add_sparql_query(&mut self, query: impl Into<String>) -> ChatResult<()> {
        let query_str = query.into();
        let (valid, error_message) = validate_sparql_query(&query_str);

        self.add_content(RichContent::SparqlQuery {
            query: query_str,
            valid,
            error_message,
            execution_plan: None,
        });

        Ok(())
    }

    /// Add a graph visualization
    pub fn add_graph_visualization(
        &mut self,
        nodes: Vec<GraphNode>,
        edges: Vec<GraphEdge>,
        layout: impl Into<String>,
    ) {
        self.add_content(RichContent::GraphVisualization {
            nodes,
            edges,
            layout: layout.into(),
            metadata: HashMap::new(),
        });
    }

    /// Add a table
    pub fn add_table(&mut self, headers: Vec<String>, rows: Vec<Vec<String>>) {
        self.add_content(RichContent::Table {
            headers,
            rows,
            metadata: HashMap::new(),
        });
    }

    /// Convert to markdown representation
    pub fn to_markdown(&self) -> String {
        let mut markdown = String::new();

        for content in &self.content_blocks {
            match content {
                RichContent::Text(text) => {
                    markdown.push_str(text);
                    markdown.push_str("\n\n");
                }
                RichContent::CodeSnippet { code, language, .. } => {
                    markdown.push_str(&format!("```{}\n{}\n```\n\n", language, code));
                }
                RichContent::SparqlQuery {
                    query,
                    valid,
                    error_message,
                    ..
                } => {
                    markdown.push_str("```sparql\n");
                    markdown.push_str(query);
                    markdown.push_str("\n```\n");

                    if !valid {
                        if let Some(error) = error_message {
                            markdown.push_str(&format!("⚠️ **Error**: {}\n", error));
                        }
                    } else {
                        markdown.push_str("✅ **Valid SPARQL query**\n");
                    }
                    markdown.push_str("\n");
                }
                RichContent::GraphVisualization { nodes, edges, .. } => {
                    markdown.push_str(&format!(
                        "📊 **Graph**: {} nodes, {} edges\n\n",
                        nodes.len(),
                        edges.len()
                    ));
                }
                RichContent::Table { headers, rows, .. } => {
                    // Convert to markdown table
                    if !headers.is_empty() {
                        markdown.push_str(&format!("| {} |\n", headers.join(" | ")));
                        markdown
                            .push_str(&format!("| {} |\n", vec!["---"; headers.len()].join(" | ")));

                        for row in rows {
                            markdown.push_str(&format!("| {} |\n", row.join(" | ")));
                        }
                        markdown.push_str("\n");
                    }
                }
                RichContent::Image { url, alt_text, .. } => {
                    let alt = alt_text.as_deref().unwrap_or("Image");
                    markdown.push_str(&format!("![{}]({})\n\n", alt, url));
                }
                RichContent::File { filename, .. } => {
                    markdown.push_str(&format!("📎 **File**: {}\n\n", filename));
                }
            }
        }

        markdown
    }

    /// Convert to HTML representation
    pub fn to_html(&self) -> String {
        let mut html = String::new();

        for content in &self.content_blocks {
            match content {
                RichContent::Text(text) => {
                    html.push_str(&format!("<p>{}</p>", html_escape(text)));
                }
                RichContent::CodeSnippet { code, language, .. } => {
                    html.push_str(&format!(
                        "<pre><code class=\"language-{}\">{}</code></pre>",
                        language,
                        html_escape(code)
                    ));
                }
                RichContent::SparqlQuery {
                    query,
                    valid,
                    error_message,
                    ..
                } => {
                    html.push_str("<div class=\"sparql-query\">");
                    html.push_str(&format!(
                        "<pre><code class=\"language-sparql\">{}</code></pre>",
                        html_escape(query)
                    ));

                    if !valid {
                        if let Some(error) = error_message {
                            html.push_str(&format!(
                                "<div class=\"error\">⚠️ <strong>Error</strong>: {}</div>",
                                html_escape(error)
                            ));
                        }
                    } else {
                        html.push_str(
                            "<div class=\"success\">✅ <strong>Valid SPARQL query</strong></div>",
                        );
                    }
                    html.push_str("</div>");
                }
                RichContent::GraphVisualization { nodes, edges, .. } => {
                    html.push_str(&format!(
                        "<div class=\"graph-visualization\">📊 <strong>Graph</strong>: {} nodes, {} edges</div>",
                        nodes.len(),
                        edges.len()
                    ));
                }
                RichContent::Table { headers, rows, .. } => {
                    html.push_str("<table>");
                    if !headers.is_empty() {
                        html.push_str("<thead><tr>");
                        for header in headers {
                            html.push_str(&format!("<th>{}</th>", html_escape(header)));
                        }
                        html.push_str("</tr></thead>");

                        html.push_str("<tbody>");
                        for row in rows {
                            html.push_str("<tr>");
                            for cell in row {
                                html.push_str(&format!("<td>{}</td>", html_escape(cell)));
                            }
                            html.push_str("</tr>");
                        }
                        html.push_str("</tbody>");
                    }
                    html.push_str("</table>");
                }
                RichContent::Image {
                    url,
                    alt_text,
                    width,
                    height,
                } => {
                    let alt = alt_text.as_deref().unwrap_or("Image");
                    let mut img_tag = format!("<img src=\"{}\" alt=\"{}\"", url, html_escape(alt));

                    if let Some(w) = width {
                        img_tag.push_str(&format!(" width=\"{}\"", w));
                    }
                    if let Some(h) = height {
                        img_tag.push_str(&format!(" height=\"{}\"", h));
                    }
                    img_tag.push_str(" />");
                    html.push_str(&img_tag);
                }
                RichContent::File { filename, .. } => {
                    html.push_str(&format!(
                        "<div class=\"file-attachment\">📎 <strong>File</strong>: {}</div>",
                        html_escape(filename)
                    ));
                }
            }
        }

        html
    }
}

impl Default for RichMessage {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate a SPARQL query
fn validate_sparql_query(query: &str) -> (bool, Option<String>) {
    // Basic SPARQL validation - check for common patterns
    let query_lower = query.to_lowercase();

    // Check if it contains basic SPARQL keywords
    let has_sparql_keywords = query_lower.contains("select")
        || query_lower.contains("construct")
        || query_lower.contains("ask")
        || query_lower.contains("describe")
        || query_lower.contains("insert")
        || query_lower.contains("delete");

    if !has_sparql_keywords {
        return (
            false,
            Some("Query does not contain SPARQL keywords".to_string()),
        );
    }

    // Check for balanced braces
    let open_braces = query.chars().filter(|&c| c == '{').count();
    let close_braces = query.chars().filter(|&c| c == '}').count();

    if open_braces != close_braces {
        return (false, Some("Unbalanced braces in query".to_string()));
    }

    // Basic syntax checks could be expanded here
    (true, None)
}

/// Escape HTML special characters
fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

/// Rich content processor for transforming and enhancing content
pub struct RichContentProcessor {
    pub syntax_highlighter_enabled: bool,
    pub graph_layout_engine: String,
    pub max_file_size: u64,
}

impl RichContentProcessor {
    /// Create a new rich content processor
    pub fn new() -> Self {
        Self {
            syntax_highlighter_enabled: true,
            graph_layout_engine: "force-directed".to_string(),
            max_file_size: 10 * 1024 * 1024, // 10MB
        }
    }

    /// Process a rich message and enhance its content
    pub fn process_message(&self, message: &mut RichMessage) -> ChatResult<()> {
        for content in &mut message.content_blocks {
            match content {
                RichContent::SparqlQuery {
                    query,
                    valid,
                    error_message,
                    execution_plan,
                } => {
                    let (is_valid, error) = validate_sparql_query(query);
                    *valid = is_valid;
                    *error_message = error;

                    if *valid {
                        *execution_plan = Some(self.generate_execution_plan(query)?);
                    }
                }
                RichContent::GraphVisualization {
                    nodes,
                    edges,
                    layout,
                    ..
                } => {
                    if layout.is_empty() {
                        *layout = self.graph_layout_engine.clone();
                    }
                    self.apply_graph_layout(nodes, edges, layout)?;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Generate an execution plan for a SPARQL query
    fn generate_execution_plan(&self, query: &str) -> ChatResult<String> {
        // This would integrate with the actual SPARQL query planner
        // For now, return a basic analysis
        let query_lower = query.to_lowercase();

        let mut plan = String::new();

        if query_lower.contains("select") {
            plan.push_str("1. Parse SELECT query\n");
            plan.push_str("2. Build graph pattern\n");
            plan.push_str("3. Execute joins\n");
            plan.push_str("4. Apply filters\n");
            plan.push_str("5. Project results\n");
        } else if query_lower.contains("construct") {
            plan.push_str("1. Parse CONSTRUCT query\n");
            plan.push_str("2. Build graph pattern\n");
            plan.push_str("3. Execute pattern matching\n");
            plan.push_str("4. Construct result graph\n");
        }

        Ok(plan)
    }

    /// Apply layout to graph visualization
    fn apply_graph_layout(
        &self,
        nodes: &mut [GraphNode],
        _edges: &[GraphEdge],
        layout: &str,
    ) -> ChatResult<()> {
        match layout {
            "force-directed" => {
                // Simple circular layout for demonstration
                let n = nodes.len() as f64;
                for (i, node) in nodes.iter_mut().enumerate() {
                    let angle = 2.0 * std::f64::consts::PI * i as f64 / n;
                    node.x = Some(50.0 + 40.0 * angle.cos());
                    node.y = Some(50.0 + 40.0 * angle.sin());
                }
            }
            "hierarchical" => {
                // Simple vertical layout
                for (i, node) in nodes.iter_mut().enumerate() {
                    node.x = Some(50.0);
                    node.y = Some(i as f64 * 20.0);
                }
            }
            _ => {
                return Err(ChatError::Processing(format!(
                    "Unsupported graph layout: {}",
                    layout
                )));
            }
        }

        Ok(())
    }
}

impl Default for RichContentProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rich_message_creation() {
        let mut message = RichMessage::new();
        message.add_text("Hello, world!");
        message.add_code("fn main() { println!(\"Hello!\"); }", "rust");

        assert_eq!(message.content_blocks.len(), 2);

        let markdown = message.to_markdown();
        assert!(markdown.contains("Hello, world!"));
        assert!(markdown.contains("```rust"));
    }

    #[test]
    fn test_sparql_validation() {
        let (valid, error) = validate_sparql_query("SELECT ?s WHERE { ?s ?p ?o }");
        assert!(valid);
        assert!(error.is_none());

        let (valid, error) = validate_sparql_query("invalid query");
        assert!(!valid);
        assert!(error.is_some());
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("<script>"), "&lt;script&gt;");
        assert_eq!(html_escape("&amp;"), "&amp;amp;");
    }

    #[test]
    fn test_graph_visualization() {
        let mut message = RichMessage::new();

        let nodes = vec![GraphNode {
            id: "node1".to_string(),
            label: "Node 1".to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            position: None,
            styling: NodeStyling {
                color: "#FF0000".to_string(),
                border_color: "#000000".to_string(),
                border_width: 1.0,
                opacity: 1.0,
                font_size: 12.0,
                font_color: "#000000".to_string(),
            },
            size: 10.0,
            shape: NodeShape::Circle,
        }];

        let edges = vec![];

        let layout = GraphLayout::ForceDirected;
        let styling = GraphStyling {
            background_color: "#FFFFFF".to_string(),
            grid_enabled: false,
            physics_enabled: true,
            clustering_enabled: false,
            smooth_curves: true,
        };
        let interactive_features = InteractiveFeatures {
            pan_enabled: true,
            zoom_enabled: true,
            node_selection: true,
            edge_selection: true,
            hover_effects: true,
            click_to_expand: false,
            context_menu: true,
        };

        message.add_content(RichContent::GraphVisualization {
            nodes,
            edges,
            layout,
            styling,
            interactive_features,
            metadata: HashMap::new(),
        });

        assert_eq!(message.content_blocks.len(), 1);

        let markdown = message.to_markdown();
        assert!(markdown.contains("Graph"));
    }
}

// Advanced visualization type definitions for Version 1.1 features

// Table types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableHeader {
    pub name: String,
    pub data_type: TableDataType,
    pub sortable: bool,
    pub filterable: bool,
    pub width: Option<u32>,
    pub alignment: TextAlignment,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableRow {
    pub cells: Vec<TableCell>,
    pub metadata: HashMap<String, String>,
    pub styling: Option<RowStyling>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableCell {
    pub value: String,
    pub data_type: TableDataType,
    pub formatting: Option<CellFormatting>,
    pub link: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TableDataType {
    Text,
    Number,
    Date,
    Boolean,
    Image,
    Link,
    Currency,
    Percentage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TextAlignment {
    Left,
    Center,
    Right,
    Justify,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TablePagination {
    pub page_size: usize,
    pub current_page: usize,
    pub total_pages: usize,
    pub show_page_numbers: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableSorting {
    pub column_index: usize,
    pub direction: SortDirection,
    pub secondary_sorts: Vec<(usize, SortDirection)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SortDirection {
    Ascending,
    Descending,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableFiltering {
    pub filters: Vec<ColumnFilter>,
    pub global_filter: Option<String>,
    pub case_sensitive: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnFilter {
    pub column_index: usize,
    pub filter_type: FilterType,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FilterType {
    Contains,
    Equals,
    StartsWith,
    EndsWith,
    GreaterThan,
    LessThan,
    Range(String, String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableStyling {
    pub theme: TableTheme,
    pub striped_rows: bool,
    pub border_style: BorderStyle,
    pub header_styling: HeaderStyling,
    pub cell_padding: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TableTheme {
    Default,
    Minimal,
    Modern,
    Classic,
    Dark,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BorderStyle {
    None,
    Solid,
    Dashed,
    Dotted,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeaderStyling {
    pub background_color: String,
    pub text_color: String,
    pub font_weight: FontWeight,
    pub fixed_header: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FontWeight {
    Normal,
    Bold,
    Light,
    ExtraBold,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RowStyling {
    pub background_color: Option<String>,
    pub text_color: Option<String>,
    pub highlighted: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellFormatting {
    pub color: Option<String>,
    pub background_color: Option<String>,
    pub font_weight: Option<FontWeight>,
    pub italic: bool,
    pub underline: bool,
}

// Chart types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChartType {
    Line,
    Bar,
    Pie,
    Scatter,
    Area,
    Histogram,
    BoxPlot,
    Radar,
    Heatmap,
    Treemap,
    Sunburst,
    Sankey,
    Gantt,
    Candlestick,
    Bubble,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChartData {
    pub datasets: Vec<ChartDataset>,
    pub categories: Vec<String>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChartDataset {
    pub label: String,
    pub data: Vec<f64>,
    pub color: String,
    pub secondary_data: Option<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChartConfiguration {
    pub title: Option<String>,
    pub width: u32,
    pub height: u32,
    pub responsive: bool,
    pub animations_enabled: bool,
    pub legend_position: LegendPosition,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LegendPosition {
    Top,
    Bottom,
    Left,
    Right,
    Hidden,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChartStyling {
    pub theme: ChartTheme,
    pub color_palette: Vec<String>,
    pub grid_lines: bool,
    pub background_color: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChartTheme {
    Default,
    Dark,
    Minimal,
    Colorful,
    Professional,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChartInteractivity {
    pub hover_enabled: bool,
    pub click_enabled: bool,
    pub zoom_enabled: bool,
    pub pan_enabled: bool,
    pub tooltip_enabled: bool,
    pub crossfilter_enabled: bool,
}

// Timeline types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelineEvent {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    pub category: String,
    pub importance: EventImportance,
    pub styling: EventStyling,
    pub attachments: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EventImportance {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventStyling {
    pub color: String,
    pub icon: Option<String>,
    pub size: EventSize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EventSize {
    Small,
    Medium,
    Large,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelineRange {
    pub start: chrono::DateTime<chrono::Utc>,
    pub end: chrono::DateTime<chrono::Utc>,
    pub default_view: TimelineView,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TimelineView {
    Day,
    Week,
    Month,
    Year,
    Decade,
    Custom(std::time::Duration),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelineStyling {
    pub orientation: TimelineOrientation,
    pub background_color: String,
    pub axis_color: String,
    pub show_grid: bool,
    pub compact_mode: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TimelineOrientation {
    Horizontal,
    Vertical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelineZoomLevel {
    pub level: u8,
    pub unit: TimeUnit,
    pub granularity: std::time::Duration,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TimeUnit {
    Millisecond,
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Year,
}

// Map types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MapType {
    Road,
    Satellite,
    Hybrid,
    Terrain,
    OpenStreetMap,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoCoordinate {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MapMarker {
    pub id: String,
    pub position: GeoCoordinate,
    pub label: String,
    pub description: Option<String>,
    pub icon: MarkerIcon,
    pub popup_content: Option<String>,
    pub clickable: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarkerIcon {
    pub icon_type: IconType,
    pub color: String,
    pub size: IconSize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IconType {
    Pin,
    Circle,
    Square,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IconSize {
    Small,
    Medium,
    Large,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MapLayer {
    pub id: String,
    pub layer_type: LayerType,
    pub data_source: String,
    pub visible: bool,
    pub opacity: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LayerType {
    Markers,
    Heatmap,
    Polygons,
    Polylines,
    Circles,
    GeoJSON,
    Tiles,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MapStyling {
    pub theme: MapTheme,
    pub show_controls: bool,
    pub show_scale: bool,
    pub show_attribution: bool,
    pub custom_style: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MapTheme {
    Default,
    Dark,
    Light,
    Satellite,
    Custom(String),
}

// 3D Visualization types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreeDObject {
    pub id: String,
    pub object_type: ObjectType,
    pub position: Vector3D,
    pub rotation: Vector3D,
    pub scale: Vector3D,
    pub material_id: String,
    pub animation_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ObjectType {
    Cube,
    Sphere,
    Cylinder,
    Plane,
    Mesh,
    PointCloud,
    Model(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Vector3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraSettings {
    pub position: Vector3D,
    pub target: Vector3D,
    pub field_of_view: f64,
    pub near_plane: f64,
    pub far_plane: f64,
    pub camera_type: CameraType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CameraType {
    Perspective,
    Orthographic,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LightingSettings {
    pub ambient_light: LightSource,
    pub directional_lights: Vec<LightSource>,
    pub point_lights: Vec<LightSource>,
    pub spot_lights: Vec<LightSource>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LightSource {
    pub id: String,
    pub light_type: LightType,
    pub color: String,
    pub intensity: f64,
    pub position: Option<Vector3D>,
    pub direction: Option<Vector3D>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LightType {
    Ambient,
    Directional,
    Point,
    Spot,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Material {
    pub id: String,
    pub material_type: MaterialType,
    pub color: String,
    pub texture: Option<String>,
    pub metallic: f64,
    pub roughness: f64,
    pub opacity: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MaterialType {
    Basic,
    Phong,
    PhysicallyBased,
    Wireframe,
    Points,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Animation {
    pub id: String,
    pub target_object_id: String,
    pub animation_type: AnimationType,
    pub duration: std::time::Duration,
    pub loop_enabled: bool,
    pub keyframes: Vec<Keyframe>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnimationType {
    Position,
    Rotation,
    Scale,
    Color,
    Opacity,
    Morph,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Keyframe {
    pub time: f64,
    pub value: Vector3D,
    pub interpolation: InterpolationType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InterpolationType {
    Linear,
    Bezier,
    Step,
}

// Dashboard types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DashboardWidget {
    pub id: String,
    pub widget_type: DashboardWidgetType,
    pub title: String,
    pub position: WidgetPosition,
    pub size: WidgetSize,
    pub data_source: String,
    pub configuration: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DashboardWidgetType {
    Chart,
    Table,
    Map,
    Text,
    Metric,
    Image,
    IFrame,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WidgetPosition {
    pub x: u32,
    pub y: u32,
    pub z_index: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WidgetSize {
    pub width: u32,
    pub height: u32,
    pub min_width: u32,
    pub min_height: u32,
    pub resizable: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DashboardLayout {
    Grid,
    Flexible,
    Masonry,
    Custom,
}

// Multimedia types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageAnnotation {
    pub id: String,
    pub annotation_type: AnnotationType,
    pub position: AnnotationPosition,
    pub content: String,
    pub styling: AnnotationStyling,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnnotationType {
    Text,
    Arrow,
    Rectangle,
    Circle,
    Highlight,
    Blur,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnotationPosition {
    pub x: f64,
    pub y: f64,
    pub width: Option<f64>,
    pub height: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnotationStyling {
    pub color: String,
    pub background_color: Option<String>,
    pub border_width: f64,
    pub font_size: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImageFilter {
    Blur(f64),
    Brightness(f64),
    Contrast(f64),
    Saturation(f64),
    Sepia(f64),
    Grayscale,
    Invert,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubtitleTrack {
    pub language: String,
    pub label: String,
    pub url: String,
    pub default: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoChapter {
    pub title: String,
    pub start_time: std::time::Duration,
    pub end_time: std::time::Duration,
    pub thumbnail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FilePreview {
    Text(String),
    Image(String),
    Audio(String),
    Video(String),
    Document(String),
    None,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileMetadata {
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub modified_at: chrono::DateTime<chrono::Utc>,
    pub author: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub custom_properties: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WidgetType {
    Button,
    Slider,
    Toggle,
    Input,
    Dropdown,
    DatePicker,
    ColorPicker,
    FileUpload,
    RangeSlider,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WidgetConfig {
    pub interactive: bool,
    pub disabled: bool,
    pub validation_rules: Vec<ValidationRule>,
    pub event_handlers: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationRule {
    pub rule_type: ValidationType,
    pub value: String,
    pub error_message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationType {
    Required,
    MinLength,
    MaxLength,
    Pattern,
    Range,
    Email,
    Url,
    Custom(String),
}
