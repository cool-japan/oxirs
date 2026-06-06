//! Rich message rendering and content processing.
//!
//! Sibling module of [`crate::rich_content`]. Provides:
//!
//! - [`RichMessage`] — a composite message made of [`RichContent`] blocks,
//!   with `to_markdown`/`to_html` rendering and ergonomic constructors.
//! - [`RichContentProcessor`] — server-side pass that validates SPARQL
//!   blocks, attaches an execution-plan sketch, and assigns layout
//!   coordinates to graph visualisations.
//! - SPARQL validation and HTML-escape helpers used by the rest of the
//!   `rich_content` family.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::rich_content_types::{
    ChatError, ChatResult, CodeTheme, GraphEdge, GraphLayout, GraphNode, GraphStyling,
    InteractiveFeatures, NodePosition, RichContent, TableCell, TableDataType, TableHeader,
    TableRow, TableStyling, TextAlignment,
};

/// Rich content message that can contain multiple content blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RichMessage {
    pub content_blocks: Vec<RichContent>,
    pub metadata: HashMap<String, String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl RichMessage {
    /// Create a new rich message.
    pub fn new() -> Self {
        Self {
            content_blocks: Vec::new(),
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Add a content block to the message.
    pub fn add_content(&mut self, content: RichContent) {
        self.content_blocks.push(content);
    }

    /// Add a text block.
    pub fn add_text(&mut self, text: impl Into<String>) {
        self.add_content(RichContent::Text(text.into()));
    }

    /// Add a code snippet.
    pub fn add_code(&mut self, code: impl Into<String>, language: impl Into<String>) {
        self.add_content(RichContent::CodeSnippet {
            code: code.into(),
            language: language.into(),
            filename: None,
            line_numbers: true,
            highlight_lines: Vec::new(),
            theme: CodeTheme::default(),
        });
    }

    /// Add a SPARQL query block.
    pub fn add_sparql_query(&mut self, query: impl Into<String>) -> ChatResult<()> {
        let query_str = query.into();
        let (valid, error_message) = validate_sparql_query(&query_str);

        self.add_content(RichContent::SparqlQuery {
            query: query_str,
            valid,
            error_message,
            execution_plan: None,
            performance_stats: None,
        });

        Ok(())
    }

    /// Add a graph visualization.
    pub fn add_graph_visualization(
        &mut self,
        nodes: Vec<GraphNode>,
        edges: Vec<GraphEdge>,
        _layout: impl Into<String>,
    ) {
        self.add_content(RichContent::GraphVisualization {
            nodes,
            edges,
            layout: GraphLayout::ForceDirected,
            styling: GraphStyling::default(),
            interactive_features: InteractiveFeatures::default(),
            metadata: HashMap::new(),
        });
    }

    /// Add a table.
    pub fn add_table(&mut self, headers: Vec<String>, rows: Vec<Vec<String>>) {
        self.add_content(RichContent::Table {
            headers: headers
                .into_iter()
                .map(|h| TableHeader {
                    name: h,
                    data_type: TableDataType::Text,
                    sortable: true,
                    filterable: true,
                    width: None,
                    alignment: TextAlignment::Left,
                })
                .collect(),
            rows: rows
                .into_iter()
                .map(|r| TableRow {
                    cells: r
                        .into_iter()
                        .map(|c| TableCell {
                            value: c,
                            data_type: TableDataType::Text,
                            formatting: None,
                            link: None,
                        })
                        .collect(),
                    metadata: HashMap::new(),
                    styling: None,
                })
                .collect(),
            pagination: None,
            sorting: None,
            filtering: None,
            styling: TableStyling::default(),
            metadata: HashMap::new(),
        });
    }

    /// Convert to markdown representation.
    pub fn to_markdown(&self) -> String {
        let mut markdown = String::new();

        for content in &self.content_blocks {
            match content {
                RichContent::Text(text) => {
                    markdown.push_str(text);
                    markdown.push_str("\n\n");
                }
                RichContent::CodeSnippet { code, language, .. } => {
                    markdown.push_str(&format!("```{language}\n{code}\n```\n\n"));
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
                            markdown.push_str(&format!("⚠️ **Error**: {error}\n"));
                        }
                    } else {
                        markdown.push_str("✅ **Valid SPARQL query**\n");
                    }
                    markdown.push('\n');
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
                        markdown.push_str(&format!(
                            "| {} |\n",
                            headers
                                .iter()
                                .map(|h| h.name.as_str())
                                .collect::<Vec<_>>()
                                .join(" | ")
                        ));
                        markdown
                            .push_str(&format!("| {} |\n", vec!["---"; headers.len()].join(" | ")));

                        for row in rows {
                            markdown.push_str(&format!(
                                "| {} |\n",
                                row.cells
                                    .iter()
                                    .map(|c| c.value.as_str())
                                    .collect::<Vec<_>>()
                                    .join(" | ")
                            ));
                        }
                        markdown.push('\n');
                    }
                }
                RichContent::Image { url, alt_text, .. } => {
                    let alt = alt_text.as_deref().unwrap_or("Image");
                    markdown.push_str(&format!("![{alt}]({url})\n\n"));
                }
                RichContent::File { filename, .. } => {
                    markdown.push_str(&format!("📎 **File**: {filename}\n\n"));
                }
                RichContent::Chart { chart_type, .. } => {
                    markdown.push_str(&format!("📈 **Chart**: {chart_type:?}\n\n"));
                }
                RichContent::Timeline { events, .. } => {
                    markdown.push_str(&format!("📅 **Timeline**: {} events\n\n", events.len()));
                }
                RichContent::Map {
                    map_type, markers, ..
                } => {
                    markdown.push_str(&format!(
                        "🗺️ **Map**: {:?} with {} markers\n\n",
                        map_type,
                        markers.len()
                    ));
                }
                RichContent::ThreeDVisualization { objects, .. } => {
                    markdown.push_str(&format!(
                        "🎯 **3D Visualization**: {} objects\n\n",
                        objects.len()
                    ));
                }
                RichContent::Dashboard { widgets, .. } => {
                    markdown.push_str(&format!("📊 **Dashboard**: {} widgets\n\n", widgets.len()));
                }
                RichContent::Audio { .. } => {
                    markdown.push_str("🔊 **Audio Content**\n\n");
                }
                RichContent::Video { .. } => {
                    markdown.push_str("🎥 **Video Content**\n\n");
                }
                RichContent::Widget { .. } => {
                    markdown.push_str("🔧 **Interactive Widget**\n\n");
                }
            }
        }

        markdown
    }

    /// Convert to HTML representation.
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
                            html.push_str(&format!("<th>{}</th>", html_escape(&header.name)));
                        }
                        html.push_str("</tr></thead>");

                        html.push_str("<tbody>");
                        for row in rows {
                            html.push_str("<tr>");
                            for cell in &row.cells {
                                html.push_str(&format!("<td>{}</td>", html_escape(&cell.value)));
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
                    annotations: _,
                    filters: _,
                } => {
                    let alt = alt_text.as_deref().unwrap_or("Image");
                    let mut img_tag = format!("<img src=\"{}\" alt=\"{}\"", url, html_escape(alt));

                    if let Some(w) = width {
                        img_tag.push_str(&format!(" width=\"{w}\""));
                    }
                    if let Some(h) = height {
                        img_tag.push_str(&format!(" height=\"{h}\""));
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
                RichContent::Chart { chart_type, .. } => {
                    html.push_str(&format!(
                        "<div class=\"chart\">📈 <strong>Chart</strong>: {chart_type:?}</div>"
                    ));
                }
                RichContent::Timeline { events, .. } => {
                    html.push_str(&format!(
                        "<div class=\"timeline\">📅 <strong>Timeline</strong>: {} events</div>",
                        events.len()
                    ));
                }
                RichContent::Map {
                    map_type, markers, ..
                } => {
                    html.push_str(&format!(
                        "<div class=\"map\">🗺️ <strong>Map</strong>: {:?} with {} markers</div>",
                        map_type,
                        markers.len()
                    ));
                }
                RichContent::ThreeDVisualization { objects, .. } => {
                    html.push_str(&format!(
                        "<div class=\"3d-viz\">🎯 <strong>3D Visualization</strong>: {} objects</div>",
                        objects.len()
                    ));
                }
                RichContent::Dashboard { widgets, .. } => {
                    html.push_str(&format!(
                        "<div class=\"dashboard\">📊 <strong>Dashboard</strong>: {} widgets</div>",
                        widgets.len()
                    ));
                }
                RichContent::Audio { .. } => {
                    html.push_str("<div class=\"audio\">🔊 <strong>Audio Content</strong></div>");
                }
                RichContent::Video { .. } => {
                    html.push_str("<div class=\"video\">🎥 <strong>Video Content</strong></div>");
                }
                RichContent::Widget { .. } => {
                    html.push_str(
                        "<div class=\"widget\">🔧 <strong>Interactive Widget</strong></div>",
                    );
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

/// Validate a SPARQL query.
pub(crate) fn validate_sparql_query(query: &str) -> (bool, Option<String>) {
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

/// Escape HTML special characters.
pub(crate) fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

/// Rich content processor for transforming and enhancing content.
pub struct RichContentProcessor {
    pub syntax_highlighter_enabled: bool,
    pub graph_layout_engine: String,
    pub max_file_size: u64,
}

impl RichContentProcessor {
    /// Create a new rich content processor.
    pub fn new() -> Self {
        Self {
            syntax_highlighter_enabled: true,
            graph_layout_engine: "force-directed".to_string(),
            max_file_size: 10 * 1024 * 1024, // 10MB
        }
    }

    /// Process a rich message and enhance its content.
    pub fn process_message(&self, message: &mut RichMessage) -> ChatResult<()> {
        for content in &mut message.content_blocks {
            match content {
                RichContent::SparqlQuery {
                    query,
                    valid,
                    error_message,
                    execution_plan,
                    performance_stats: _,
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
                    let layout_str = match layout {
                        GraphLayout::ForceDirected => "force-directed",
                        GraphLayout::Circular => "circular",
                        GraphLayout::Hierarchical => "hierarchical",
                        GraphLayout::Grid => "grid",
                        GraphLayout::Tree => "tree",
                        GraphLayout::Cluster => "cluster",
                        GraphLayout::Custom { algorithm, .. } => algorithm,
                    };
                    self.apply_graph_layout(nodes, edges, layout_str)?;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Generate an execution plan for a SPARQL query.
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

    /// Apply layout to graph visualization.
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
                    node.position = Some(NodePosition {
                        x: 50.0 + 40.0 * angle.cos(),
                        y: 50.0 + 40.0 * angle.sin(),
                        z: None,
                    });
                }
            }
            "hierarchical" => {
                // Simple vertical layout
                for (i, node) in nodes.iter_mut().enumerate() {
                    node.position = Some(NodePosition {
                        x: 50.0,
                        y: i as f64 * 20.0,
                        z: None,
                    });
                }
            }
            _ => {
                return Err(ChatError::Processing(format!(
                    "Unsupported graph layout: {layout}"
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
