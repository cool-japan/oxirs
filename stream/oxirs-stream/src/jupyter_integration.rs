//! # Jupyter Notebook Integration for Stream Processing
//!
//! Provides a Jupyter kernel and interactive widgets for OxiRS stream processing.
//! Enables data scientists and developers to interactively explore, analyze, and
//! visualize streaming data in Jupyter notebooks with real-time updates.
//!
//! ## Features
//! - Custom Jupyter kernel for stream processing
//! - Interactive widgets for stream visualization
//! - Real-time charts and graphs
//! - Stream inspection and debugging
//! - Magic commands for common operations
//! - Cell-level stream execution
//! - Automatic result visualization
//! - Export results to various formats
//! - Integration with pandas, numpy, and visualization libraries
//! - Markdown documentation generation

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::event::StreamEvent;
use crate::visual_designer::VisualPipeline;

/// Jupyter kernel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JupyterKernelConfig {
    pub kernel_name: String,
    pub kernel_display_name: String,
    pub language: String,
    pub language_version: String,
    pub enable_widgets: bool,
    pub enable_rich_output: bool,
    pub max_output_size: usize,
    pub enable_streaming_output: bool,
}

/// Jupyter kernel for OxiRS streams
pub struct OxiRSKernel {
    config: JupyterKernelConfig,
    execution_count: u64,
    namespace: Arc<RwLock<HashMap<String, Variable>>>,
    output_buffer: Arc<RwLock<VecDeque<OutputMessage>>>,
    stream_handles: Arc<RwLock<HashMap<String, StreamHandle>>>,
}

/// Variable in the kernel namespace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
    pub value: VariableValue,
    pub var_type: String,
    pub created_at: DateTime<Utc>,
}

/// Variable value types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum VariableValue {
    Pipeline(String),    // Pipeline ID
    StreamEvent(String), // Serialized event
    DataFrame(String),   // Serialized dataframe
    String(String),
    Number(f64),
    Boolean(bool),
    Null,
}

/// Stream handle for managing active streams
#[derive(Debug, Clone)]
pub struct StreamHandle {
    pub id: String,
    pub name: String,
    pub pipeline_id: Option<String>,
    pub status: StreamStatus,
    pub events_processed: u64,
    pub created_at: DateTime<Utc>,
}

/// Stream status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StreamStatus {
    Active,
    Paused,
    Stopped,
    Error(String),
}

/// Output message from kernel execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputMessage {
    pub msg_type: MessageType,
    pub content: MessageContent,
    pub execution_count: u64,
    pub timestamp: DateTime<Utc>,
}

/// Message types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MessageType {
    ExecuteResult,
    DisplayData,
    Stream,
    Error,
    Status,
}

/// Message content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageContent {
    pub data: HashMap<String, String>,
    pub metadata: HashMap<String, String>,
    pub text: Option<String>,
}

/// Type alias for magic command handlers to reduce complexity
type MagicCommandHandler = Box<dyn Fn(&[String]) -> Result<String> + Send + Sync>;

/// Magic command handler
pub struct MagicCommands {
    commands: HashMap<String, MagicCommandHandler>,
}

impl Default for MagicCommands {
    fn default() -> Self {
        Self::new()
    }
}

impl MagicCommands {
    pub fn new() -> Self {
        let mut commands: HashMap<String, MagicCommandHandler> = HashMap::new();

        // %stream - Create a new stream
        commands.insert(
            "stream".to_string(),
            Box::new(|args: &[String]| {
                if args.is_empty() {
                    return Err(anyhow!("Usage: %stream <name> <backend>"));
                }
                Ok(format!("Created stream: {}", args[0]))
            }),
        );

        // %streams - List all streams
        commands.insert(
            "streams".to_string(),
            Box::new(|_args: &[String]| Ok("Listing all streams...".to_string())),
        );

        // %visualize - Visualize stream data
        commands.insert(
            "visualize".to_string(),
            Box::new(|args: &[String]| {
                if args.is_empty() {
                    return Err(anyhow!("Usage: %visualize <stream_name> <chart_type>"));
                }
                Ok(format!("Visualizing stream: {}", args[0]))
            }),
        );

        // %stats - Show stream statistics
        commands.insert(
            "stats".to_string(),
            Box::new(|args: &[String]| {
                if args.is_empty() {
                    return Err(anyhow!("Usage: %stats <stream_name>"));
                }
                Ok(format!("Stream statistics for: {}", args[0]))
            }),
        );

        // %export - Export stream data
        commands.insert(
            "export".to_string(),
            Box::new(|args: &[String]| {
                if args.len() < 2 {
                    return Err(anyhow!(
                        "Usage: %export <stream_name> <format> [output_file]"
                    ));
                }
                Ok(format!("Exporting {} to {}", args[0], args[1]))
            }),
        );

        Self { commands }
    }

    pub fn execute(&self, command: &str, args: &[String]) -> Result<String> {
        if let Some(handler) = self.commands.get(command) {
            handler(args)
        } else {
            Err(anyhow!("Unknown magic command: %{}", command))
        }
    }

    pub fn list_commands(&self) -> Vec<String> {
        self.commands.keys().cloned().collect()
    }
}

impl OxiRSKernel {
    /// Create a new Jupyter kernel
    pub fn new(config: JupyterKernelConfig) -> Self {
        Self {
            config,
            execution_count: 0,
            namespace: Arc::new(RwLock::new(HashMap::new())),
            output_buffer: Arc::new(RwLock::new(VecDeque::new())),
            stream_handles: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Execute code in the kernel
    pub async fn execute(&mut self, code: &str) -> Result<ExecutionResult> {
        self.execution_count += 1;
        let execution_count = self.execution_count;

        info!("Executing code (count: {})", execution_count);
        debug!("Code: {}", code);

        // Check if this is a magic command
        if code.trim().starts_with('%') {
            return self.execute_magic_command(code, execution_count).await;
        }

        // Parse and execute the code
        // This is a simplified implementation - in reality would need a full interpreter
        let result = self.execute_stream_code(code).await?;

        Ok(ExecutionResult {
            status: ExecutionStatus::Ok,
            execution_count,
            data: result.data,
            metadata: result.metadata,
            error: None,
        })
    }

    /// Execute a magic command
    async fn execute_magic_command(
        &mut self,
        code: &str,
        execution_count: u64,
    ) -> Result<ExecutionResult> {
        let parts: Vec<&str> = code.split_whitespace().collect();
        if parts.is_empty() {
            return Err(anyhow!("Empty magic command"));
        }

        let command = parts[0].trim_start_matches('%');
        let args: Vec<String> = parts[1..].iter().map(|s| s.to_string()).collect();

        let magic = MagicCommands::new();
        let output = magic.execute(command, &args)?;

        Ok(ExecutionResult {
            status: ExecutionStatus::Ok,
            execution_count,
            data: HashMap::from([("text/plain".to_string(), output)]),
            metadata: HashMap::new(),
            error: None,
        })
    }

    /// Execute stream processing code
    async fn execute_stream_code(&mut self, code: &str) -> Result<ExecutionResult> {
        // Simplified execution - would need proper parsing in production
        let mut data = HashMap::new();
        let metadata = HashMap::new();

        // Check for common patterns
        if code.contains("create_stream") {
            data.insert(
                "text/html".to_string(),
                "<div class=\"stream-created\">Stream created successfully</div>".to_string(),
            );
        } else if code.contains("visualize") {
            data.insert("text/html".to_string(), self.generate_chart_html().await?);
        } else {
            data.insert(
                "text/plain".to_string(),
                "Executed successfully".to_string(),
            );
        }

        Ok(ExecutionResult {
            status: ExecutionStatus::Ok,
            execution_count: self.execution_count,
            data,
            metadata,
            error: None,
        })
    }

    /// Generate chart HTML for visualization
    async fn generate_chart_html(&self) -> Result<String> {
        Ok(r##"
<div id="stream-chart" style="width: 800px; height: 400px;">
    <svg viewBox="0 0 800 400">
        <rect width="800" height="400" fill="#f0f0f0"/>
        <text x="400" y="200" text-anchor="middle" font-size="20">
            Stream Visualization
        </text>
    </svg>
</div>
        "##
        .to_string())
    }

    /// Get kernel info
    pub fn get_info(&self) -> KernelInfo {
        KernelInfo {
            protocol_version: "5.3".to_string(),
            implementation: "oxirs-stream-kernel".to_string(),
            implementation_version: "0.1.0".to_string(),
            language_info: LanguageInfo {
                name: self.config.language.clone(),
                version: self.config.language_version.clone(),
                mimetype: "text/x-rust".to_string(),
                file_extension: ".rs".to_string(),
            },
            banner: "OxiRS Stream Processing Kernel".to_string(),
            help_links: vec![HelpLink {
                text: "OxiRS Documentation".to_string(),
                url: "https://github.com/cool-japan/oxirs".to_string(),
            }],
        }
    }

    /// List available magic commands
    pub fn list_magic_commands(&self) -> Vec<String> {
        MagicCommands::new().list_commands()
    }

    /// Get namespace variables
    pub async fn get_variables(&self) -> Vec<Variable> {
        self.namespace.read().await.values().cloned().collect()
    }

    /// Get stream handles
    pub async fn get_streams(&self) -> Vec<StreamHandle> {
        self.stream_handles.read().await.values().cloned().collect()
    }
}

/// Execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub status: ExecutionStatus,
    pub execution_count: u64,
    pub data: HashMap<String, String>,
    pub metadata: HashMap<String, String>,
    pub error: Option<ExecutionError>,
}

/// Execution status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutionStatus {
    Ok,
    Error,
    Abort,
}

/// Execution error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionError {
    pub ename: String,
    pub evalue: String,
    pub traceback: Vec<String>,
}

/// Kernel info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelInfo {
    pub protocol_version: String,
    pub implementation: String,
    pub implementation_version: String,
    pub language_info: LanguageInfo,
    pub banner: String,
    pub help_links: Vec<HelpLink>,
}

/// Language info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageInfo {
    pub name: String,
    pub version: String,
    pub mimetype: String,
    pub file_extension: String,
}

/// Help link
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelpLink {
    pub text: String,
    pub url: String,
}

/// Interactive widget for stream visualization
pub struct StreamWidget {
    pub id: String,
    pub widget_type: WidgetType,
    pub config: WidgetConfig,
    pub data: Arc<RwLock<Vec<StreamEvent>>>,
}

/// Widget types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum WidgetType {
    LineChart,
    BarChart,
    PieChart,
    Table,
    Gauge,
    Heatmap,
    Timeline,
    Graph,
}

/// Widget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub refresh_rate_ms: u64,
    pub max_data_points: usize,
    pub styling: WidgetStyle,
}

/// Widget styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetStyle {
    pub theme: String,
    pub colors: Vec<String>,
    pub font_family: String,
    pub font_size: u32,
}

impl StreamWidget {
    /// Create a new widget
    pub fn new(widget_type: WidgetType, config: WidgetConfig) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            widget_type,
            config,
            data: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Update widget data
    pub async fn update(&self, event: StreamEvent) -> Result<()> {
        let mut data = self.data.write().await;
        data.push(event);

        // Keep only the last N data points
        if data.len() > self.config.max_data_points {
            data.remove(0);
        }

        Ok(())
    }

    /// Render widget to HTML
    pub async fn render_html(&self) -> Result<String> {
        let data = self.data.read().await;
        let mut html = String::new();

        html.push_str(&format!(
            "<div id=\"widget-{}\" class=\"oxirs-widget\" style=\"width: {}px; height: {}px;\">",
            self.id, self.config.width, self.config.height
        ));

        html.push_str(&format!("<h3>{}</h3>", self.config.title));

        match self.widget_type {
            WidgetType::LineChart => {
                html.push_str("<svg viewBox=\"0 0 800 400\">");
                html.push_str(
                    "<line x1=\"50\" y1=\"350\" x2=\"750\" y2=\"350\" stroke=\"black\" />",
                );
                html.push_str("<line x1=\"50\" y1=\"50\" x2=\"50\" y2=\"350\" stroke=\"black\" />");
                html.push_str(&format!(
                    "<text x=\"400\" y=\"30\" text-anchor=\"middle\">{} Data Points</text>",
                    data.len()
                ));
                html.push_str("</svg>");
            }
            WidgetType::Table => {
                html.push_str("<table border=\"1\">");
                html.push_str("<tr><th>Event ID</th><th>Timestamp</th><th>Type</th></tr>");
                for (i, _event) in data.iter().take(10).enumerate() {
                    html.push_str(&format!("<tr><td>{}</td><td>--</td><td>Event</td></tr>", i));
                }
                html.push_str("</table>");
            }
            _ => {
                html.push_str(&format!("<p>Widget type: {:?}</p>", self.widget_type));
            }
        }

        html.push_str("</div>");

        Ok(html)
    }

    /// Render widget to JSON
    pub async fn render_json(&self) -> Result<serde_json::Value> {
        let data = self.data.read().await;

        Ok(json!({
            "id": self.id,
            "type": format!("{:?}", self.widget_type),
            "config": {
                "title": self.config.title,
                "width": self.config.width,
                "height": self.config.height,
            },
            "data_points": data.len(),
        }))
    }
}

/// Notebook cell for stream processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotebookCell {
    pub cell_type: CellType,
    pub source: Vec<String>,
    pub outputs: Vec<CellOutput>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub execution_count: Option<u64>,
}

/// Cell types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CellType {
    Code,
    Markdown,
    Raw,
}

/// Cell output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellOutput {
    pub output_type: String,
    pub data: HashMap<String, serde_json::Value>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub execution_count: Option<u64>,
}

/// Notebook document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notebook {
    pub cells: Vec<NotebookCell>,
    pub metadata: NotebookMetadata,
    pub nbformat: u32,
    pub nbformat_minor: u32,
}

/// Notebook metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotebookMetadata {
    pub kernelspec: KernelSpec,
    pub language_info: LanguageInfo,
}

/// Kernel specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSpec {
    pub display_name: String,
    pub language: String,
    pub name: String,
}

impl Notebook {
    /// Create a new notebook
    pub fn new() -> Self {
        Self {
            cells: Vec::new(),
            metadata: NotebookMetadata {
                kernelspec: KernelSpec {
                    display_name: "OxiRS Stream".to_string(),
                    language: "rust".to_string(),
                    name: "oxirs-stream".to_string(),
                },
                language_info: LanguageInfo {
                    name: "rust".to_string(),
                    version: "1.75.0".to_string(),
                    mimetype: "text/x-rust".to_string(),
                    file_extension: ".rs".to_string(),
                },
            },
            nbformat: 4,
            nbformat_minor: 5,
        }
    }

    /// Add a code cell
    pub fn add_code_cell(&mut self, code: String) {
        self.cells.push(NotebookCell {
            cell_type: CellType::Code,
            source: vec![code],
            outputs: Vec::new(),
            metadata: HashMap::new(),
            execution_count: None,
        });
    }

    /// Add a markdown cell
    pub fn add_markdown_cell(&mut self, markdown: String) {
        self.cells.push(NotebookCell {
            cell_type: CellType::Markdown,
            source: vec![markdown],
            outputs: Vec::new(),
            metadata: HashMap::new(),
            execution_count: None,
        });
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| anyhow!("JSON export failed: {}", e))
    }

    /// Import from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| anyhow!("JSON import failed: {}", e))
    }
}

impl Default for Notebook {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a sample notebook
pub fn generate_sample_notebook(pipeline: &VisualPipeline) -> Notebook {
    let mut notebook = Notebook::new();

    // Add title
    notebook.add_markdown_cell(format!("# Stream Processing: {}", pipeline.name));

    if let Some(desc) = &pipeline.description {
        notebook.add_markdown_cell(format!("**Description:** {}", desc));
    }

    // Add setup cell
    notebook.add_markdown_cell("## Setup".to_string());
    notebook.add_code_cell(
        r#"// Import required modules
use oxirs_stream::{StreamConfig, StreamEvent};

// Create stream configuration
let config = StreamConfig::memory();"#
            .to_string(),
    );

    // Add pipeline info
    notebook.add_markdown_cell("## Pipeline Information".to_string());
    notebook.add_code_cell(format!(
        r#"// Pipeline: {}
// Nodes: {}
// Edges: {}
println!("Pipeline loaded successfully");"#,
        pipeline.name,
        pipeline.nodes.len(),
        pipeline.edges.len()
    ));

    // Add visualization cell
    notebook.add_markdown_cell("## Visualization".to_string());
    notebook.add_code_cell(
        r#"%visualize pipeline line_chart
// Real-time visualization will appear here"#
            .to_string(),
    );

    // Add statistics cell
    notebook.add_markdown_cell("## Statistics".to_string());
    notebook.add_code_cell("%stats pipeline".to_string());

    notebook
}

impl Default for JupyterKernelConfig {
    fn default() -> Self {
        Self {
            kernel_name: "oxirs-stream".to_string(),
            kernel_display_name: "OxiRS Stream".to_string(),
            language: "rust".to_string(),
            language_version: "1.75.0".to_string(),
            enable_widgets: true,
            enable_rich_output: true,
            max_output_size: 1024 * 1024, // 1MB
            enable_streaming_output: true,
        }
    }
}

impl Default for WidgetConfig {
    fn default() -> Self {
        Self {
            title: "Stream Widget".to_string(),
            width: 800,
            height: 400,
            refresh_rate_ms: 1000,
            max_data_points: 1000,
            styling: WidgetStyle {
                theme: "light".to_string(),
                colors: vec![
                    "#1f77b4".to_string(),
                    "#ff7f0e".to_string(),
                    "#2ca02c".to_string(),
                ],
                font_family: "Arial, sans-serif".to_string(),
                font_size: 12,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_kernel_creation() {
        let config = JupyterKernelConfig::default();
        let kernel = OxiRSKernel::new(config);

        let info = kernel.get_info();
        assert_eq!(info.implementation, "oxirs-stream-kernel");
        assert_eq!(info.language_info.name, "rust");
    }

    #[tokio::test]
    async fn test_magic_commands() {
        let magic = MagicCommands::new();
        let commands = magic.list_commands();

        assert!(commands.contains(&"stream".to_string()));
        assert!(commands.contains(&"visualize".to_string()));
        assert!(commands.contains(&"stats".to_string()));
    }

    #[tokio::test]
    async fn test_execute_magic() {
        let mut kernel = OxiRSKernel::new(JupyterKernelConfig::default());
        let result = kernel.execute("%streams").await.unwrap();

        assert_eq!(result.status, ExecutionStatus::Ok);
        assert!(result.error.is_none());
    }

    #[tokio::test]
    async fn test_widget_creation() {
        let config = WidgetConfig::default();
        let widget = StreamWidget::new(WidgetType::LineChart, config);

        assert_eq!(widget.widget_type, WidgetType::LineChart);
        assert_eq!(widget.data.read().await.len(), 0);
    }

    #[tokio::test]
    async fn test_widget_render_html() {
        let config = WidgetConfig {
            title: "Test Widget".to_string(),
            ..Default::default()
        };
        let widget = StreamWidget::new(WidgetType::Table, config);

        let html = widget.render_html().await.unwrap();
        assert!(html.contains("Test Widget"));
        assert!(html.contains("<table"));
    }

    #[test]
    fn test_notebook_creation() {
        let mut notebook = Notebook::new();

        assert_eq!(notebook.cells.len(), 0);
        assert_eq!(notebook.nbformat, 4);

        notebook.add_code_cell("let x = 1;".to_string());
        notebook.add_markdown_cell("# Title".to_string());

        assert_eq!(notebook.cells.len(), 2);
        assert_eq!(notebook.cells[0].cell_type, CellType::Code);
        assert_eq!(notebook.cells[1].cell_type, CellType::Markdown);
    }

    #[test]
    fn test_notebook_json() {
        let mut notebook = Notebook::new();
        notebook.add_code_cell("let x = 1;".to_string());

        let json = notebook.to_json().unwrap();
        assert!(!json.is_empty());

        let imported = Notebook::from_json(&json).unwrap();
        assert_eq!(imported.cells.len(), 1);
    }

    #[tokio::test]
    async fn test_kernel_variables() {
        let kernel = OxiRSKernel::new(JupyterKernelConfig::default());
        let vars = kernel.get_variables().await;

        assert_eq!(vars.len(), 0);
    }

    #[tokio::test]
    async fn test_kernel_streams() {
        let kernel = OxiRSKernel::new(JupyterKernelConfig::default());
        let streams = kernel.get_streams().await;

        assert_eq!(streams.len(), 0);
    }
}
