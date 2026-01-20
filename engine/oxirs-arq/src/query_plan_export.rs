//! # Query Plan Export
//!
//! This module provides functionality to export SPARQL query execution plans
//! to various formats for analysis, visualization, and debugging.
//!
//! ## Supported Formats
//!
//! - **JSON**: Machine-readable format for tooling integration
//! - **DOT (Graphviz)**: Graph visualization for plan structure
//! - **Mermaid**: Web-friendly diagram format
//! - **Text**: Human-readable text representation
//! - **YAML**: Configuration-friendly format
//! - **HTML**: Interactive web visualization
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use oxirs_arq::query_plan_export::{
//!     QueryPlanExporter, ExportFormat, PlanNode,
//! };
//!
//! // Create a plan exporter
//! let exporter = QueryPlanExporter::new();
//!
//! // Build a plan tree
//! let plan = PlanNode::scan("?s ?p ?o");
//!
//! // Export to various formats
//! let json = exporter.export(&plan, ExportFormat::Json)?;
//! let dot = exporter.export(&plan, ExportFormat::Dot)?;
//! ```

use std::collections::HashMap;
use std::fmt;

/// Supported export formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExportFormat {
    /// JSON format
    Json,
    /// DOT (Graphviz) format
    Dot,
    /// Mermaid diagram format
    Mermaid,
    /// Plain text format
    Text,
    /// YAML format
    Yaml,
    /// HTML format with embedded visualization
    Html,
}

impl fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json => write!(f, "JSON"),
            Self::Dot => write!(f, "DOT"),
            Self::Mermaid => write!(f, "Mermaid"),
            Self::Text => write!(f, "Text"),
            Self::Yaml => write!(f, "YAML"),
            Self::Html => write!(f, "HTML"),
        }
    }
}

impl ExportFormat {
    /// Get file extension for format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Dot => "dot",
            Self::Mermaid => "md",
            Self::Text => "txt",
            Self::Yaml => "yaml",
            Self::Html => "html",
        }
    }

    /// Get MIME type for format
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Json => "application/json",
            Self::Dot => "text/vnd.graphviz",
            Self::Mermaid => "text/markdown",
            Self::Text => "text/plain",
            Self::Yaml => "application/x-yaml",
            Self::Html => "text/html",
        }
    }
}

/// Plan node operator type
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OperatorType {
    /// Triple pattern scan
    Scan,
    /// Hash join
    HashJoin,
    /// Merge join
    MergeJoin,
    /// Nested loop join
    NestedLoopJoin,
    /// Index join
    IndexJoin,
    /// Filter operation
    Filter,
    /// Projection (SELECT)
    Project,
    /// Distinct operation
    Distinct,
    /// Order by operation
    OrderBy,
    /// Limit operation
    Limit,
    /// Offset operation
    Offset,
    /// Group by operation
    GroupBy,
    /// Aggregation
    Aggregate,
    /// Union operation
    Union,
    /// Optional pattern
    Optional,
    /// Minus operation
    Minus,
    /// Service (federation)
    Service,
    /// Graph pattern
    Graph,
    /// Bind operation
    Bind,
    /// Values clause
    Values,
    /// Property path
    PropertyPath,
    /// Subquery
    Subquery,
    /// Custom operator
    Custom(String),
}

impl fmt::Display for OperatorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scan => write!(f, "Scan"),
            Self::HashJoin => write!(f, "HashJoin"),
            Self::MergeJoin => write!(f, "MergeJoin"),
            Self::NestedLoopJoin => write!(f, "NestedLoopJoin"),
            Self::IndexJoin => write!(f, "IndexJoin"),
            Self::Filter => write!(f, "Filter"),
            Self::Project => write!(f, "Project"),
            Self::Distinct => write!(f, "Distinct"),
            Self::OrderBy => write!(f, "OrderBy"),
            Self::Limit => write!(f, "Limit"),
            Self::Offset => write!(f, "Offset"),
            Self::GroupBy => write!(f, "GroupBy"),
            Self::Aggregate => write!(f, "Aggregate"),
            Self::Union => write!(f, "Union"),
            Self::Optional => write!(f, "Optional"),
            Self::Minus => write!(f, "Minus"),
            Self::Service => write!(f, "Service"),
            Self::Graph => write!(f, "Graph"),
            Self::Bind => write!(f, "Bind"),
            Self::Values => write!(f, "Values"),
            Self::PropertyPath => write!(f, "PropertyPath"),
            Self::Subquery => write!(f, "Subquery"),
            Self::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Cost estimate for a plan node
#[derive(Debug, Clone, Default)]
pub struct CostEstimate {
    /// Estimated row count
    pub estimated_rows: f64,
    /// Estimated cost (abstract units)
    pub estimated_cost: f64,
    /// Estimated memory usage in bytes
    pub estimated_memory: usize,
    /// Estimated I/O operations
    pub estimated_io: usize,
}

/// Execution statistics for a plan node (actual execution)
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Actual rows processed
    pub actual_rows: usize,
    /// Actual execution time in milliseconds
    pub execution_time_ms: f64,
    /// Actual memory used in bytes
    pub memory_used: usize,
    /// Number of iterations (for nested loops)
    pub iterations: usize,
}

/// A node in the query plan tree
#[derive(Debug, Clone)]
pub struct PlanNode {
    /// Unique node ID
    pub id: String,
    /// Operator type
    pub operator: OperatorType,
    /// Human-readable description
    pub description: String,
    /// Variables involved
    pub variables: Vec<String>,
    /// Child nodes
    pub children: Vec<PlanNode>,
    /// Cost estimates
    pub cost: Option<CostEstimate>,
    /// Actual execution statistics
    pub stats: Option<ExecutionStats>,
    /// Additional properties
    pub properties: HashMap<String, String>,
}

impl PlanNode {
    /// Create a new plan node
    pub fn new(operator: OperatorType, description: impl Into<String>) -> Self {
        Self {
            id: Self::generate_id(),
            operator,
            description: description.into(),
            variables: Vec::new(),
            children: Vec::new(),
            cost: None,
            stats: None,
            properties: HashMap::new(),
        }
    }

    /// Create a scan node
    pub fn scan(pattern: impl Into<String>) -> Self {
        Self::new(OperatorType::Scan, pattern)
    }

    /// Create a hash join node
    pub fn hash_join(description: impl Into<String>) -> Self {
        Self::new(OperatorType::HashJoin, description)
    }

    /// Create a filter node
    pub fn filter(condition: impl Into<String>) -> Self {
        Self::new(OperatorType::Filter, condition)
    }

    /// Create a project node
    pub fn project(vars: impl Into<String>) -> Self {
        Self::new(OperatorType::Project, vars)
    }

    /// Create a distinct node
    pub fn distinct() -> Self {
        Self::new(OperatorType::Distinct, "DISTINCT")
    }

    /// Create an order by node
    pub fn order_by(ordering: impl Into<String>) -> Self {
        Self::new(OperatorType::OrderBy, ordering)
    }

    /// Create a limit node
    pub fn limit(n: usize) -> Self {
        Self::new(OperatorType::Limit, format!("LIMIT {}", n))
    }

    /// Create an offset node
    pub fn offset(n: usize) -> Self {
        Self::new(OperatorType::Offset, format!("OFFSET {}", n))
    }

    /// Create a union node
    pub fn union() -> Self {
        Self::new(OperatorType::Union, "UNION")
    }

    /// Create an optional node
    pub fn optional() -> Self {
        Self::new(OperatorType::Optional, "OPTIONAL")
    }

    /// Create a group by node
    pub fn group_by(vars: impl Into<String>) -> Self {
        Self::new(OperatorType::GroupBy, vars)
    }

    /// Create an aggregate node
    pub fn aggregate(agg: impl Into<String>) -> Self {
        Self::new(OperatorType::Aggregate, agg)
    }

    /// Add a child node
    pub fn with_child(mut self, child: PlanNode) -> Self {
        self.children.push(child);
        self
    }

    /// Add multiple children
    pub fn with_children(mut self, children: Vec<PlanNode>) -> Self {
        self.children.extend(children);
        self
    }

    /// Add variables
    pub fn with_variables(mut self, vars: Vec<String>) -> Self {
        self.variables = vars;
        self
    }

    /// Add cost estimate
    pub fn with_cost(mut self, cost: CostEstimate) -> Self {
        self.cost = Some(cost);
        self
    }

    /// Add execution stats
    pub fn with_stats(mut self, stats: ExecutionStats) -> Self {
        self.stats = Some(stats);
        self
    }

    /// Add a property
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Count total nodes in tree
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.node_count()).sum::<usize>()
    }

    /// Get tree depth
    pub fn depth(&self) -> usize {
        if self.children.is_empty() {
            1
        } else {
            1 + self.children.iter().map(|c| c.depth()).max().unwrap_or(0)
        }
    }

    fn generate_id() -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        format!("node_{}", COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// Export configuration
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Include cost estimates
    pub include_costs: bool,
    /// Include execution statistics
    pub include_stats: bool,
    /// Include node properties
    pub include_properties: bool,
    /// Include variables
    pub include_variables: bool,
    /// Pretty print output
    pub pretty_print: bool,
    /// Indentation string (for text/yaml)
    pub indent: String,
    /// Include header/metadata
    pub include_metadata: bool,
    /// Graph direction for DOT/Mermaid (TB, BT, LR, RL)
    pub graph_direction: String,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            include_costs: true,
            include_stats: true,
            include_properties: true,
            include_variables: true,
            pretty_print: true,
            indent: "  ".to_string(),
            include_metadata: true,
            graph_direction: "TB".to_string(),
        }
    }
}

impl ExportConfig {
    /// Minimal configuration
    pub fn minimal() -> Self {
        Self {
            include_costs: false,
            include_stats: false,
            include_properties: false,
            include_variables: false,
            pretty_print: false,
            indent: "  ".to_string(),
            include_metadata: false,
            graph_direction: "TB".to_string(),
        }
    }

    /// Full configuration with all details
    pub fn full() -> Self {
        Self {
            include_costs: true,
            include_stats: true,
            include_properties: true,
            include_variables: true,
            pretty_print: true,
            indent: "    ".to_string(),
            include_metadata: true,
            graph_direction: "TB".to_string(),
        }
    }
}

/// Query plan exporter
#[derive(Debug)]
pub struct QueryPlanExporter {
    /// Export configuration
    config: ExportConfig,
    /// Statistics
    stats: ExporterStats,
}

/// Exporter statistics
#[derive(Debug, Clone, Default)]
pub struct ExporterStats {
    /// Total exports performed
    pub total_exports: usize,
    /// Exports by format
    pub exports_by_format: HashMap<String, usize>,
    /// Total nodes exported
    pub total_nodes_exported: usize,
}

impl QueryPlanExporter {
    /// Create a new exporter with default configuration
    pub fn new() -> Self {
        Self {
            config: ExportConfig::default(),
            stats: ExporterStats::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ExportConfig) -> Self {
        Self {
            config,
            stats: ExporterStats::default(),
        }
    }

    /// Export plan to specified format
    pub fn export(&mut self, plan: &PlanNode, format: ExportFormat) -> Result<String, ExportError> {
        self.stats.total_exports += 1;
        *self
            .stats
            .exports_by_format
            .entry(format.to_string())
            .or_insert(0) += 1;
        self.stats.total_nodes_exported += plan.node_count();

        match format {
            ExportFormat::Json => self.export_json(plan),
            ExportFormat::Dot => self.export_dot(plan),
            ExportFormat::Mermaid => self.export_mermaid(plan),
            ExportFormat::Text => self.export_text(plan),
            ExportFormat::Yaml => self.export_yaml(plan),
            ExportFormat::Html => self.export_html(plan),
        }
    }

    /// Get exporter statistics
    pub fn statistics(&self) -> &ExporterStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &ExportConfig {
        &self.config
    }

    // Private export methods

    fn export_json(&self, plan: &PlanNode) -> Result<String, ExportError> {
        let mut output = String::new();

        if self.config.pretty_print {
            self.json_node_pretty(&mut output, plan, 0);
        } else {
            self.json_node(&mut output, plan);
        }

        Ok(output)
    }

    fn json_node(&self, output: &mut String, node: &PlanNode) {
        output.push('{');
        output.push_str(&format!("\"id\":\"{}\"", node.id));
        output.push_str(&format!(",\"operator\":\"{}\"", node.operator));
        output.push_str(&format!(
            ",\"description\":\"{}\"",
            Self::escape_json(&node.description)
        ));

        if self.config.include_variables && !node.variables.is_empty() {
            output.push_str(",\"variables\":[");
            for (i, var) in node.variables.iter().enumerate() {
                if i > 0 {
                    output.push(',');
                }
                output.push_str(&format!("\"{}\"", var));
            }
            output.push(']');
        }

        if self.config.include_costs {
            if let Some(ref cost) = node.cost {
                output.push_str(&format!(
                    ",\"cost\":{{\"estimated_rows\":{},\"estimated_cost\":{}}}",
                    cost.estimated_rows, cost.estimated_cost
                ));
            }
        }

        if self.config.include_stats {
            if let Some(ref stats) = node.stats {
                output.push_str(&format!(
                    ",\"stats\":{{\"actual_rows\":{},\"execution_time_ms\":{}}}",
                    stats.actual_rows, stats.execution_time_ms
                ));
            }
        }

        if !node.children.is_empty() {
            output.push_str(",\"children\":[");
            for (i, child) in node.children.iter().enumerate() {
                if i > 0 {
                    output.push(',');
                }
                self.json_node(output, child);
            }
            output.push(']');
        }

        output.push('}');
    }

    fn json_node_pretty(&self, output: &mut String, node: &PlanNode, depth: usize) {
        let indent = self.config.indent.repeat(depth);
        let child_indent = self.config.indent.repeat(depth + 1);

        output.push_str(&format!("{}{{\n", indent));
        output.push_str(&format!("{}\"id\": \"{}\",\n", child_indent, node.id));
        output.push_str(&format!(
            "{}\"operator\": \"{}\",\n",
            child_indent, node.operator
        ));
        output.push_str(&format!(
            "{}\"description\": \"{}\"",
            child_indent,
            Self::escape_json(&node.description)
        ));

        if self.config.include_variables && !node.variables.is_empty() {
            output.push_str(&format!(
                ",\n{}\"variables\": {:?}",
                child_indent, node.variables
            ));
        }

        if self.config.include_costs {
            if let Some(ref cost) = node.cost {
                output.push_str(&format!(
                    ",\n{}\"cost\": {{\n{}\"estimated_rows\": {},\n{}\"estimated_cost\": {}\n{}}}",
                    child_indent,
                    self.config.indent.repeat(depth + 2),
                    cost.estimated_rows,
                    self.config.indent.repeat(depth + 2),
                    cost.estimated_cost,
                    child_indent
                ));
            }
        }

        if !node.children.is_empty() {
            output.push_str(&format!(",\n{}\"children\": [\n", child_indent));
            for (i, child) in node.children.iter().enumerate() {
                if i > 0 {
                    output.push_str(",\n");
                }
                self.json_node_pretty(output, child, depth + 2);
            }
            output.push_str(&format!("\n{}]", child_indent));
        }

        output.push_str(&format!("\n{}}}", indent));
    }

    fn export_dot(&self, plan: &PlanNode) -> Result<String, ExportError> {
        let mut output = String::new();

        output.push_str("digraph QueryPlan {\n");
        output.push_str(&format!(
            "{}rankdir={};\n",
            self.config.indent, self.config.graph_direction
        ));
        output.push_str(&format!(
            "{}node [shape=box, style=rounded];\n",
            self.config.indent
        ));

        self.dot_node(&mut output, plan);

        output.push_str("}\n");
        Ok(output)
    }

    fn dot_node(&self, output: &mut String, node: &PlanNode) {
        let label = self.dot_label(node);
        output.push_str(&format!(
            "{}\"{}\" [label=\"{}\"];\n",
            self.config.indent, node.id, label
        ));

        for child in &node.children {
            output.push_str(&format!(
                "{}\"{}\" -> \"{}\";\n",
                self.config.indent, node.id, child.id
            ));
            self.dot_node(output, child);
        }
    }

    fn dot_label(&self, node: &PlanNode) -> String {
        let mut label = format!("{}\\n{}", node.operator, node.description);

        if self.config.include_costs {
            if let Some(ref cost) = node.cost {
                label.push_str(&format!(
                    "\\n[rows: {:.0}, cost: {:.2}]",
                    cost.estimated_rows, cost.estimated_cost
                ));
            }
        }

        if self.config.include_stats {
            if let Some(ref stats) = node.stats {
                label.push_str(&format!(
                    "\\n(actual: {} rows, {:.2}ms)",
                    stats.actual_rows, stats.execution_time_ms
                ));
            }
        }

        label
    }

    fn export_mermaid(&self, plan: &PlanNode) -> Result<String, ExportError> {
        let mut output = String::new();

        output.push_str("```mermaid\n");
        output.push_str(&format!("graph {}\n", self.config.graph_direction));

        self.mermaid_node(&mut output, plan);

        output.push_str("```\n");
        Ok(output)
    }

    fn mermaid_node(&self, output: &mut String, node: &PlanNode) {
        let label = self.mermaid_label(node);
        output.push_str(&format!(
            "{}{}[\"{}\"]\n",
            self.config.indent, node.id, label
        ));

        for child in &node.children {
            output.push_str(&format!(
                "{}{} --> {}\n",
                self.config.indent, node.id, child.id
            ));
            self.mermaid_node(output, child);
        }
    }

    fn mermaid_label(&self, node: &PlanNode) -> String {
        let mut label = format!("{}: {}", node.operator, node.description);

        if self.config.include_costs {
            if let Some(ref cost) = node.cost {
                label.push_str(&format!(" [rows: {:.0}]", cost.estimated_rows));
            }
        }

        label
    }

    fn export_text(&self, plan: &PlanNode) -> Result<String, ExportError> {
        let mut output = String::new();

        if self.config.include_metadata {
            output.push_str("Query Plan\n");
            output.push_str(&format!("Nodes: {}\n", plan.node_count()));
            output.push_str(&format!("Depth: {}\n", plan.depth()));
            output.push_str("─".repeat(40).as_str());
            output.push('\n');
        }

        self.text_node(&mut output, plan, 0);
        Ok(output)
    }

    fn text_node(&self, output: &mut String, node: &PlanNode, depth: usize) {
        let prefix = if depth == 0 {
            "".to_string()
        } else {
            format!("{}├── ", "│   ".repeat(depth - 1))
        };

        output.push_str(&format!(
            "{}{}: {}",
            prefix, node.operator, node.description
        ));

        if self.config.include_costs {
            if let Some(ref cost) = node.cost {
                output.push_str(&format!(
                    " (rows: {:.0}, cost: {:.2})",
                    cost.estimated_rows, cost.estimated_cost
                ));
            }
        }

        if self.config.include_stats {
            if let Some(ref stats) = node.stats {
                output.push_str(&format!(
                    " [actual: {} rows, {:.2}ms]",
                    stats.actual_rows, stats.execution_time_ms
                ));
            }
        }

        output.push('\n');

        for child in &node.children {
            self.text_node(output, child, depth + 1);
        }
    }

    fn export_yaml(&self, plan: &PlanNode) -> Result<String, ExportError> {
        let mut output = String::new();

        if self.config.include_metadata {
            output.push_str("# Query Plan Export\n");
            output.push_str(&format!("# Nodes: {}\n", plan.node_count()));
            output.push_str(&format!("# Depth: {}\n\n", plan.depth()));
        }

        self.yaml_node(&mut output, plan, 0);
        Ok(output)
    }

    fn yaml_node(&self, output: &mut String, node: &PlanNode, depth: usize) {
        let indent = self.config.indent.repeat(depth);

        output.push_str(&format!("{}id: {}\n", indent, node.id));
        output.push_str(&format!("{}operator: {}\n", indent, node.operator));
        output.push_str(&format!(
            "{}description: \"{}\"\n",
            indent, node.description
        ));

        if self.config.include_variables && !node.variables.is_empty() {
            output.push_str(&format!("{}variables:\n", indent));
            for var in &node.variables {
                output.push_str(&format!(
                    "{}- {}\n",
                    self.config.indent.repeat(depth + 1),
                    var
                ));
            }
        }

        if self.config.include_costs {
            if let Some(ref cost) = node.cost {
                output.push_str(&format!("{}cost:\n", indent));
                output.push_str(&format!(
                    "{}estimated_rows: {}\n",
                    self.config.indent.repeat(depth + 1),
                    cost.estimated_rows
                ));
                output.push_str(&format!(
                    "{}estimated_cost: {}\n",
                    self.config.indent.repeat(depth + 1),
                    cost.estimated_cost
                ));
            }
        }

        if !node.children.is_empty() {
            output.push_str(&format!("{}children:\n", indent));
            for child in &node.children {
                output.push_str(&format!("{}- \n", self.config.indent.repeat(depth + 1)));
                self.yaml_node(output, child, depth + 2);
            }
        }
    }

    fn export_html(&self, plan: &PlanNode) -> Result<String, ExportError> {
        let mut output = String::new();

        output.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        output.push_str("<title>Query Plan</title>\n");
        output.push_str("<style>\n");
        output.push_str(
            ".node { border: 1px solid #ccc; border-radius: 8px; padding: 10px; margin: 5px; background: #f9f9f9; }\n",
        );
        output.push_str(".operator { font-weight: bold; color: #333; }\n");
        output.push_str(".description { color: #666; font-family: monospace; }\n");
        output.push_str(".cost { color: #999; font-size: 0.9em; }\n");
        output.push_str(".children { margin-left: 20px; border-left: 2px solid #ddd; }\n");
        output.push_str("</style>\n</head>\n<body>\n");

        if self.config.include_metadata {
            output.push_str("<h1>Query Plan</h1>\n");
            output.push_str(&format!(
                "<p>Nodes: {} | Depth: {}</p>\n",
                plan.node_count(),
                plan.depth()
            ));
        }

        self.html_node(&mut output, plan);

        output.push_str("</body>\n</html>\n");
        Ok(output)
    }

    fn html_node(&self, output: &mut String, node: &PlanNode) {
        output.push_str("<div class=\"node\">\n");
        output.push_str(&format!(
            "<span class=\"operator\">{}</span>: ",
            node.operator
        ));
        output.push_str(&format!(
            "<span class=\"description\">{}</span>\n",
            Self::escape_html(&node.description)
        ));

        if self.config.include_costs {
            if let Some(ref cost) = node.cost {
                output.push_str(&format!(
                    "<div class=\"cost\">Est. rows: {:.0}, Cost: {:.2}</div>\n",
                    cost.estimated_rows, cost.estimated_cost
                ));
            }
        }

        if !node.children.is_empty() {
            output.push_str("<div class=\"children\">\n");
            for child in &node.children {
                self.html_node(output, child);
            }
            output.push_str("</div>\n");
        }

        output.push_str("</div>\n");
    }

    fn escape_json(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
    }

    fn escape_html(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    }
}

impl Default for QueryPlanExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Export error
#[derive(Debug, Clone)]
pub struct ExportError {
    /// Error message
    pub message: String,
    /// Format that caused the error
    pub format: ExportFormat,
}

impl fmt::Display for ExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Export error ({}): {}", self.format, self.message)
    }
}

impl std::error::Error for ExportError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_node_creation() {
        let node = PlanNode::scan("?s ?p ?o");
        assert_eq!(node.operator, OperatorType::Scan);
        assert_eq!(node.description, "?s ?p ?o");
    }

    #[test]
    fn test_plan_node_with_children() {
        let plan = PlanNode::hash_join("?s = ?s")
            .with_child(PlanNode::scan("?s :knows ?o"))
            .with_child(PlanNode::scan("?s :name ?n"));

        assert_eq!(plan.children.len(), 2);
        assert_eq!(plan.node_count(), 3);
        assert_eq!(plan.depth(), 2);
    }

    #[test]
    fn test_plan_node_with_cost() {
        let cost = CostEstimate {
            estimated_rows: 100.0,
            estimated_cost: 50.0,
            ..Default::default()
        };
        let node = PlanNode::scan("?s ?p ?o").with_cost(cost);

        assert!(node.cost.is_some());
        assert_eq!(node.cost.as_ref().unwrap().estimated_rows, 100.0);
    }

    #[test]
    fn test_export_json() {
        let plan = PlanNode::scan("?s ?p ?o");
        let mut exporter = QueryPlanExporter::new();

        let json = exporter.export(&plan, ExportFormat::Json).unwrap();
        // JSON may be pretty-printed with spaces after colons
        assert!(json.contains("\"operator\""));
        assert!(json.contains("Scan"));
        assert!(json.contains("?s ?p ?o"));
    }

    #[test]
    fn test_export_dot() {
        let plan = PlanNode::hash_join("join").with_child(PlanNode::scan("?s ?p ?o"));

        let mut exporter = QueryPlanExporter::new();
        let dot = exporter.export(&plan, ExportFormat::Dot).unwrap();

        assert!(dot.contains("digraph QueryPlan"));
        assert!(dot.contains("->"));
    }

    #[test]
    fn test_export_mermaid() {
        let plan = PlanNode::scan("?s ?p ?o");
        let mut exporter = QueryPlanExporter::new();

        let mermaid = exporter.export(&plan, ExportFormat::Mermaid).unwrap();
        assert!(mermaid.contains("```mermaid"));
        assert!(mermaid.contains("graph TB"));
    }

    #[test]
    fn test_export_text() {
        let plan = PlanNode::project("?s ?o").with_child(PlanNode::scan("?s ?p ?o"));

        let mut exporter = QueryPlanExporter::new();
        let text = exporter.export(&plan, ExportFormat::Text).unwrap();

        assert!(text.contains("Project"));
        assert!(text.contains("Scan"));
    }

    #[test]
    fn test_export_yaml() {
        let plan = PlanNode::scan("?s ?p ?o");
        let mut exporter = QueryPlanExporter::new();

        let yaml = exporter.export(&plan, ExportFormat::Yaml).unwrap();
        assert!(yaml.contains("operator: Scan"));
    }

    #[test]
    fn test_export_html() {
        let plan = PlanNode::scan("?s ?p ?o");
        let mut exporter = QueryPlanExporter::new();

        let html = exporter.export(&plan, ExportFormat::Html).unwrap();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Scan"));
    }

    #[test]
    fn test_export_format_extension() {
        assert_eq!(ExportFormat::Json.extension(), "json");
        assert_eq!(ExportFormat::Dot.extension(), "dot");
        assert_eq!(ExportFormat::Html.extension(), "html");
    }

    #[test]
    fn test_export_format_mime_type() {
        assert_eq!(ExportFormat::Json.mime_type(), "application/json");
        assert_eq!(ExportFormat::Html.mime_type(), "text/html");
    }

    #[test]
    fn test_exporter_statistics() {
        let mut exporter = QueryPlanExporter::new();
        let plan = PlanNode::scan("?s ?p ?o");

        exporter.export(&plan, ExportFormat::Json).unwrap();
        exporter.export(&plan, ExportFormat::Dot).unwrap();

        assert_eq!(exporter.statistics().total_exports, 2);
        assert_eq!(exporter.statistics().total_nodes_exported, 2);
    }

    #[test]
    fn test_config_presets() {
        let minimal = ExportConfig::minimal();
        assert!(!minimal.include_costs);
        assert!(!minimal.include_stats);

        let full = ExportConfig::full();
        assert!(full.include_costs);
        assert!(full.include_stats);
    }

    #[test]
    fn test_operator_types() {
        assert_eq!(format!("{}", OperatorType::Scan), "Scan");
        assert_eq!(format!("{}", OperatorType::HashJoin), "HashJoin");
        assert_eq!(
            format!("{}", OperatorType::Custom("MyOp".to_string())),
            "MyOp"
        );
    }

    #[test]
    fn test_complex_plan() {
        let plan = PlanNode::project("?name ?age")
            .with_child(
                PlanNode::filter("?age > 18").with_child(
                    PlanNode::hash_join("?person = ?person")
                        .with_child(PlanNode::scan("?person :name ?name"))
                        .with_child(PlanNode::scan("?person :age ?age")),
                ),
            )
            .with_cost(CostEstimate {
                estimated_rows: 50.0,
                estimated_cost: 120.0,
                ..Default::default()
            });

        assert_eq!(plan.node_count(), 5);
        assert_eq!(plan.depth(), 4);

        let mut exporter = QueryPlanExporter::new();
        let json = exporter.export(&plan, ExportFormat::Json).unwrap();
        assert!(json.contains("Project"));
        assert!(json.contains("Filter"));
        assert!(json.contains("HashJoin"));
    }
}
