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
    },
    /// SPARQL query block with validation
    SparqlQuery {
        query: String,
        valid: bool,
        error_message: Option<String>,
        execution_plan: Option<String>,
    },
    /// Graph visualization data
    GraphVisualization {
        nodes: Vec<GraphNode>,
        edges: Vec<GraphEdge>,
        layout: String,
        metadata: HashMap<String, String>,
    },
    /// Table output with headers and rows
    Table {
        headers: Vec<String>,
        rows: Vec<Vec<String>>,
        metadata: HashMap<String, String>,
    },
    /// Image attachment
    Image {
        url: String,
        alt_text: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
    },
    /// File upload
    File {
        path: PathBuf,
        filename: String,
        mime_type: String,
        size: u64,
    },
}

/// Graph node for visualization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub node_type: String,
    pub properties: HashMap<String, String>,
    pub x: Option<f64>,
    pub y: Option<f64>,
}

/// Graph edge for visualization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub label: String,
    pub edge_type: String,
    pub properties: HashMap<String, String>,
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
            x: None,
            y: None,
        }];

        let edges = vec![];

        message.add_graph_visualization(nodes, edges, "force-directed");

        assert_eq!(message.content_blocks.len(), 1);

        let markdown = message.to_markdown();
        assert!(markdown.contains("Graph"));
    }
}
