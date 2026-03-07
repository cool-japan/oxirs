//! # SPARQL Query Plan Visualizer
//!
//! Renders SPARQL query execution plans in multiple formats:
//! - **DOT / Graphviz**: for `dot`-based graph rendering
//! - **Text tree**: human-readable indented plan tree
//! - **JSON**: machine-readable plan representation
//!
//! Displays join order, estimated costs, filter placement, and operator types.
//!
//! ## Quick Start
//!
//! ```rust
//! use oxirs_arq::plan_visualizer::{
//!     QueryPlanVisualizer, VisPlanNode, VisOperator, VisOutputFormat,
//! };
//!
//! let leaf = VisPlanNode::leaf(VisOperator::Scan, "?s ?p ?o")
//!     .with_estimated_rows(1000)
//!     .with_cost(5.0);
//! let root = VisPlanNode::unary(VisOperator::Projection, "?s ?o", leaf)
//!     .with_estimated_rows(1000)
//!     .with_cost(6.0);
//!
//! let viz = QueryPlanVisualizer::new();
//! let dot = viz.render(&root, VisOutputFormat::Dot).unwrap();
//! assert!(dot.contains("digraph"));
//! ```

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Operator types
// ---------------------------------------------------------------------------

/// Operator types that appear in a query plan.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VisOperator {
    /// Triple / quad pattern scan
    Scan,
    /// Hash join
    HashJoin,
    /// Merge join
    MergeJoin,
    /// Nested-loop join
    NestedLoopJoin,
    /// Lateral join (SPARQL LATERAL)
    LateralJoin,
    /// Left outer join (OPTIONAL)
    LeftJoin,
    /// UNION of two branches
    Union,
    /// FILTER
    Filter,
    /// BIND expression
    Bind,
    /// ORDER BY
    Sort,
    /// DISTINCT
    Distinct,
    /// LIMIT / OFFSET
    Slice,
    /// GROUP BY aggregation
    Aggregate,
    /// Projection (SELECT columns)
    Projection,
    /// SERVICE (federated)
    Service,
    /// Sub-query
    SubQuery,
    /// VALUES injection
    Values,
    /// GRAPH pattern
    Graph,
    /// Materialised intermediate result
    Materialise,
    /// Custom / user-defined operator
    Custom(String),
}

impl fmt::Display for VisOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scan => write!(f, "Scan"),
            Self::HashJoin => write!(f, "HashJoin"),
            Self::MergeJoin => write!(f, "MergeJoin"),
            Self::NestedLoopJoin => write!(f, "NestedLoopJoin"),
            Self::LateralJoin => write!(f, "LateralJoin"),
            Self::LeftJoin => write!(f, "LeftJoin"),
            Self::Union => write!(f, "Union"),
            Self::Filter => write!(f, "Filter"),
            Self::Bind => write!(f, "Bind"),
            Self::Sort => write!(f, "Sort"),
            Self::Distinct => write!(f, "Distinct"),
            Self::Slice => write!(f, "Slice"),
            Self::Aggregate => write!(f, "Aggregate"),
            Self::Projection => write!(f, "Projection"),
            Self::Service => write!(f, "Service"),
            Self::SubQuery => write!(f, "SubQuery"),
            Self::Values => write!(f, "Values"),
            Self::Graph => write!(f, "Graph"),
            Self::Materialise => write!(f, "Materialise"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}

impl VisOperator {
    /// Short label for DOT nodes.
    pub fn short_label(&self) -> &str {
        match self {
            Self::Scan => "Scan",
            Self::HashJoin => "HJ",
            Self::MergeJoin => "MJ",
            Self::NestedLoopJoin => "NLJ",
            Self::LateralJoin => "LAT",
            Self::LeftJoin => "LOJ",
            Self::Union => "U",
            Self::Filter => "F",
            Self::Bind => "B",
            Self::Sort => "Sort",
            Self::Distinct => "Dist",
            Self::Slice => "Slice",
            Self::Aggregate => "Agg",
            Self::Projection => "Pi",
            Self::Service => "Svc",
            Self::SubQuery => "SQ",
            Self::Values => "Val",
            Self::Graph => "G",
            Self::Materialise => "Mat",
            Self::Custom(_) => "Cust",
        }
    }

    /// Whether this is a join operator.
    pub fn is_join(&self) -> bool {
        matches!(
            self,
            Self::HashJoin
                | Self::MergeJoin
                | Self::NestedLoopJoin
                | Self::LateralJoin
                | Self::LeftJoin
        )
    }
}

// ---------------------------------------------------------------------------
// Plan node
// ---------------------------------------------------------------------------

/// A node in the visualisable query plan tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisPlanNode {
    /// Unique ID (auto-assigned during rendering if zero).
    pub id: u64,
    /// The operator at this node.
    pub operator: VisOperator,
    /// Human-readable description / pattern.
    pub description: String,
    /// Estimated output rows (cardinality).
    pub estimated_rows: Option<u64>,
    /// Estimated cost (arbitrary units, higher = more expensive).
    pub estimated_cost: Option<f64>,
    /// Actual rows (after execution; may be absent before execution).
    pub actual_rows: Option<u64>,
    /// Execution time in microseconds (may be absent before execution).
    pub execution_time_us: Option<u64>,
    /// Operator-specific properties (e.g. join condition, filter expression).
    pub properties: HashMap<String, String>,
    /// Child nodes (0 for leaves, 1 for unary, 2 for binary, etc.).
    pub children: Vec<VisPlanNode>,
}

impl VisPlanNode {
    /// Create a leaf node (no children).
    pub fn leaf(operator: VisOperator, description: impl Into<String>) -> Self {
        Self {
            id: 0,
            operator,
            description: description.into(),
            estimated_rows: None,
            estimated_cost: None,
            actual_rows: None,
            execution_time_us: None,
            properties: HashMap::new(),
            children: Vec::new(),
        }
    }

    /// Create a unary operator node (one child).
    pub fn unary(
        operator: VisOperator,
        description: impl Into<String>,
        child: VisPlanNode,
    ) -> Self {
        Self {
            id: 0,
            operator,
            description: description.into(),
            estimated_rows: None,
            estimated_cost: None,
            actual_rows: None,
            execution_time_us: None,
            properties: HashMap::new(),
            children: vec![child],
        }
    }

    /// Create a binary operator node (two children, e.g. join).
    pub fn binary(
        operator: VisOperator,
        description: impl Into<String>,
        left: VisPlanNode,
        right: VisPlanNode,
    ) -> Self {
        Self {
            id: 0,
            operator,
            description: description.into(),
            estimated_rows: None,
            estimated_cost: None,
            actual_rows: None,
            execution_time_us: None,
            properties: HashMap::new(),
            children: vec![left, right],
        }
    }

    /// Builder: set estimated rows.
    pub fn with_estimated_rows(mut self, rows: u64) -> Self {
        self.estimated_rows = Some(rows);
        self
    }

    /// Builder: set estimated cost.
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.estimated_cost = Some(cost);
        self
    }

    /// Builder: set actual rows.
    pub fn with_actual_rows(mut self, rows: u64) -> Self {
        self.actual_rows = Some(rows);
        self
    }

    /// Builder: set execution time (microseconds).
    pub fn with_execution_time_us(mut self, us: u64) -> Self {
        self.execution_time_us = Some(us);
        self
    }

    /// Builder: add an arbitrary property.
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Count the total number of nodes in the subtree rooted at this node.
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.node_count()).sum::<usize>()
    }

    /// Maximum depth of the subtree.
    pub fn depth(&self) -> usize {
        if self.children.is_empty() {
            1
        } else {
            1 + self.children.iter().map(|c| c.depth()).max().unwrap_or(0)
        }
    }

    /// Collect all filter nodes in the plan.
    pub fn collect_filters(&self) -> Vec<&VisPlanNode> {
        let mut result = Vec::new();
        if self.operator == VisOperator::Filter {
            result.push(self);
        }
        for child in &self.children {
            result.extend(child.collect_filters());
        }
        result
    }

    /// Collect all join nodes in the plan.
    pub fn collect_joins(&self) -> Vec<&VisPlanNode> {
        let mut result = Vec::new();
        if self.operator.is_join() {
            result.push(self);
        }
        for child in &self.children {
            result.extend(child.collect_joins());
        }
        result
    }

    /// Total estimated cost of the subtree (sum of all node costs).
    pub fn total_cost(&self) -> f64 {
        let self_cost = self.estimated_cost.unwrap_or(0.0);
        self_cost + self.children.iter().map(|c| c.total_cost()).sum::<f64>()
    }
}

// ---------------------------------------------------------------------------
// Output format
// ---------------------------------------------------------------------------

/// The output format for the visualiser.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VisOutputFormat {
    /// DOT / Graphviz
    Dot,
    /// Human-readable text tree
    TextTree,
    /// Machine-readable JSON
    Json,
}

impl fmt::Display for VisOutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dot => write!(f, "DOT"),
            Self::TextTree => write!(f, "TextTree"),
            Self::Json => write!(f, "JSON"),
        }
    }
}

// ---------------------------------------------------------------------------
// Visualiser configuration
// ---------------------------------------------------------------------------

/// Configuration for the plan visualiser.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizerConfig {
    /// Show estimated rows in labels.
    pub show_estimated_rows: bool,
    /// Show cost in labels.
    pub show_cost: bool,
    /// Show actual rows (post-execution).
    pub show_actual_rows: bool,
    /// Show execution time.
    pub show_execution_time: bool,
    /// Show operator properties.
    pub show_properties: bool,
    /// Use colour in DOT output.
    pub use_colour: bool,
    /// DOT graph orientation: "TB" (top-bottom) or "LR" (left-right).
    pub dot_orientation: String,
    /// Indent string for text tree (e.g. "  " or "    ").
    pub indent: String,
    /// Pretty-print JSON output.
    pub json_pretty: bool,
}

impl Default for VisualizerConfig {
    fn default() -> Self {
        Self {
            show_estimated_rows: true,
            show_cost: true,
            show_actual_rows: true,
            show_execution_time: true,
            show_properties: true,
            use_colour: true,
            dot_orientation: "TB".to_string(),
            indent: "  ".to_string(),
            json_pretty: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics summary
// ---------------------------------------------------------------------------

/// Summary statistics computed from a plan tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanSummary {
    /// Total number of operators.
    pub total_nodes: usize,
    /// Maximum depth.
    pub max_depth: usize,
    /// Number of join operators.
    pub join_count: usize,
    /// Number of filter operators.
    pub filter_count: usize,
    /// Number of scan operators.
    pub scan_count: usize,
    /// Total estimated cost.
    pub total_estimated_cost: f64,
    /// Operator type frequency.
    pub operator_histogram: HashMap<String, usize>,
}

// ---------------------------------------------------------------------------
// QueryPlanVisualizer
// ---------------------------------------------------------------------------

/// The main entry point for rendering query plans.
pub struct QueryPlanVisualizer {
    config: VisualizerConfig,
}

impl Default for QueryPlanVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryPlanVisualizer {
    /// Create a visualiser with default configuration.
    pub fn new() -> Self {
        Self {
            config: VisualizerConfig::default(),
        }
    }

    /// Create a visualiser with custom configuration.
    pub fn with_config(config: VisualizerConfig) -> Self {
        Self { config }
    }

    /// Render the plan tree in the requested format.
    pub fn render(
        &self,
        root: &VisPlanNode,
        format: VisOutputFormat,
    ) -> std::result::Result<String, String> {
        // First assign IDs by traversal order.
        let mut id_counter = 1_u64;
        let annotated = self.assign_ids(root, &mut id_counter);

        match format {
            VisOutputFormat::Dot => Ok(self.render_dot(&annotated)),
            VisOutputFormat::TextTree => Ok(self.render_text_tree(&annotated)),
            VisOutputFormat::Json => self.render_json(&annotated),
        }
    }

    /// Compute summary statistics from a plan tree.
    pub fn summarise(&self, root: &VisPlanNode) -> PlanSummary {
        let mut histogram: HashMap<String, usize> = HashMap::new();
        Self::count_operators(root, &mut histogram);

        PlanSummary {
            total_nodes: root.node_count(),
            max_depth: root.depth(),
            join_count: root.collect_joins().len(),
            filter_count: root.collect_filters().len(),
            scan_count: *histogram.get("Scan").unwrap_or(&0),
            total_estimated_cost: root.total_cost(),
            operator_histogram: histogram,
        }
    }

    // -----------------------------------------------------------------------
    // internal helpers
    // -----------------------------------------------------------------------

    fn count_operators(node: &VisPlanNode, histogram: &mut HashMap<String, usize>) {
        *histogram.entry(node.operator.to_string()).or_insert(0) += 1;
        for child in &node.children {
            Self::count_operators(child, histogram);
        }
    }

    /// Recursively clone the tree assigning monotonic IDs.
    fn assign_ids(&self, node: &VisPlanNode, counter: &mut u64) -> VisPlanNode {
        let id = *counter;
        *counter += 1;
        let children = node
            .children
            .iter()
            .map(|c| self.assign_ids(c, counter))
            .collect();
        VisPlanNode {
            id,
            operator: node.operator.clone(),
            description: node.description.clone(),
            estimated_rows: node.estimated_rows,
            estimated_cost: node.estimated_cost,
            actual_rows: node.actual_rows,
            execution_time_us: node.execution_time_us,
            properties: node.properties.clone(),
            children,
        }
    }

    // -----------------------------------------------------------------------
    // DOT rendering
    // -----------------------------------------------------------------------

    fn render_dot(&self, root: &VisPlanNode) -> String {
        let mut buf = String::new();
        buf.push_str("digraph QueryPlan {\n");
        buf.push_str(&format!("  rankdir={};\n", self.config.dot_orientation));
        buf.push_str("  node [shape=record, fontname=\"Helvetica\", fontsize=10];\n");
        buf.push_str("  edge [fontname=\"Helvetica\", fontsize=9];\n\n");

        self.dot_nodes(root, &mut buf);
        buf.push('\n');
        self.dot_edges(root, &mut buf);

        buf.push_str("}\n");
        buf
    }

    fn dot_nodes(&self, node: &VisPlanNode, buf: &mut String) {
        let label = self.dot_label(node);
        let colour = if self.config.use_colour {
            self.operator_colour(&node.operator)
        } else {
            "white"
        };
        buf.push_str(&format!(
            "  n{} [label=\"{}\", style=filled, fillcolor=\"{}\"];\n",
            node.id, label, colour
        ));
        for child in &node.children {
            self.dot_nodes(child, buf);
        }
    }

    fn dot_label(&self, node: &VisPlanNode) -> String {
        let mut parts = vec![format!(
            "{}|{}",
            node.operator.short_label(),
            Self::escape_dot(&node.description)
        )];
        if self.config.show_estimated_rows {
            if let Some(rows) = node.estimated_rows {
                parts.push(format!("est: {rows} rows"));
            }
        }
        if self.config.show_cost {
            if let Some(cost) = node.estimated_cost {
                parts.push(format!("cost: {cost:.1}"));
            }
        }
        if self.config.show_actual_rows {
            if let Some(rows) = node.actual_rows {
                parts.push(format!("actual: {rows} rows"));
            }
        }
        if self.config.show_execution_time {
            if let Some(us) = node.execution_time_us {
                parts.push(format!("time: {us} us"));
            }
        }
        if self.config.show_properties {
            for (k, v) in &node.properties {
                parts.push(format!("{k}: {}", Self::escape_dot(v)));
            }
        }
        format!("{{{}}}", parts.join("|"))
    }

    fn dot_edges(&self, node: &VisPlanNode, buf: &mut String) {
        for (i, child) in node.children.iter().enumerate() {
            let label = if node.children.len() > 1 {
                match i {
                    0 => "left",
                    1 => "right",
                    _ => "child",
                }
            } else {
                "input"
            };
            buf.push_str(&format!(
                "  n{} -> n{} [label=\"{}\"];\n",
                node.id, child.id, label
            ));
            self.dot_edges(child, buf);
        }
    }

    fn escape_dot(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('{', "\\{")
            .replace('}', "\\}")
            .replace('<', "\\<")
            .replace('>', "\\>")
            .replace('|', "\\|")
    }

    fn operator_colour(&self, op: &VisOperator) -> &'static str {
        match op {
            VisOperator::Scan => "#E8F5E9",
            VisOperator::HashJoin | VisOperator::MergeJoin | VisOperator::NestedLoopJoin => {
                "#E3F2FD"
            }
            VisOperator::LeftJoin | VisOperator::LateralJoin => "#E8EAF6",
            VisOperator::Filter => "#FFF3E0",
            VisOperator::Sort | VisOperator::Distinct | VisOperator::Slice => "#F3E5F5",
            VisOperator::Aggregate => "#FCE4EC",
            VisOperator::Union => "#E0F7FA",
            VisOperator::Projection => "#F1F8E9",
            VisOperator::Service | VisOperator::SubQuery => "#FFF9C4",
            _ => "#FAFAFA",
        }
    }

    // -----------------------------------------------------------------------
    // Text tree rendering
    // -----------------------------------------------------------------------

    fn render_text_tree(&self, root: &VisPlanNode) -> String {
        let mut buf = String::new();
        self.text_tree_node(root, &mut buf, "", true);
        buf
    }

    fn text_tree_node(&self, node: &VisPlanNode, buf: &mut String, prefix: &str, is_last: bool) {
        let connector = if prefix.is_empty() {
            ""
        } else if is_last {
            "`-- "
        } else {
            "|-- "
        };

        buf.push_str(prefix);
        buf.push_str(connector);
        buf.push_str(&format!("[{}] {}", node.operator, node.description));

        // inline stats
        let mut annotations = Vec::new();
        if self.config.show_estimated_rows {
            if let Some(rows) = node.estimated_rows {
                annotations.push(format!("est={rows}"));
            }
        }
        if self.config.show_cost {
            if let Some(cost) = node.estimated_cost {
                annotations.push(format!("cost={cost:.1}"));
            }
        }
        if self.config.show_actual_rows {
            if let Some(rows) = node.actual_rows {
                annotations.push(format!("actual={rows}"));
            }
        }
        if self.config.show_execution_time {
            if let Some(us) = node.execution_time_us {
                annotations.push(format!("time={us}us"));
            }
        }
        if !annotations.is_empty() {
            buf.push_str(&format!("  ({})", annotations.join(", ")));
        }
        buf.push('\n');

        // properties
        if self.config.show_properties {
            let child_prefix = if prefix.is_empty() {
                self.config.indent.clone()
            } else if is_last {
                format!("{prefix}{}", self.config.indent)
            } else {
                format!("{prefix}|{}", &self.config.indent[1..])
            };
            for (k, v) in &node.properties {
                buf.push_str(&format!("{child_prefix}  {k}: {v}\n"));
            }
        }

        // children
        let child_prefix = if prefix.is_empty() {
            String::new()
        } else if is_last {
            format!("{prefix}{}", self.config.indent)
        } else {
            format!("{prefix}|{}", &self.config.indent[1..])
        };
        for (i, child) in node.children.iter().enumerate() {
            let last = i == node.children.len() - 1;
            self.text_tree_node(child, buf, &child_prefix, last);
        }
    }

    // -----------------------------------------------------------------------
    // JSON rendering
    // -----------------------------------------------------------------------

    fn render_json(&self, root: &VisPlanNode) -> std::result::Result<String, String> {
        if self.config.json_pretty {
            serde_json::to_string_pretty(root).map_err(|e| e.to_string())
        } else {
            serde_json::to_string(root).map_err(|e| e.to_string())
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ------------------------------------------------------------

    fn simple_scan() -> VisPlanNode {
        VisPlanNode::leaf(VisOperator::Scan, "?s ?p ?o")
            .with_estimated_rows(1000)
            .with_cost(5.0)
    }

    fn two_scan_join() -> VisPlanNode {
        let left = VisPlanNode::leaf(VisOperator::Scan, "?s :name ?name")
            .with_estimated_rows(500)
            .with_cost(3.0);
        let right = VisPlanNode::leaf(VisOperator::Scan, "?s :age ?age")
            .with_estimated_rows(500)
            .with_cost(3.0);
        VisPlanNode::binary(VisOperator::HashJoin, "?s", left, right)
            .with_estimated_rows(200)
            .with_cost(10.0)
    }

    fn complex_plan() -> VisPlanNode {
        let scan1 = VisPlanNode::leaf(VisOperator::Scan, "?s :name ?name")
            .with_estimated_rows(1000)
            .with_cost(5.0);
        let scan2 = VisPlanNode::leaf(VisOperator::Scan, "?s :age ?age")
            .with_estimated_rows(800)
            .with_cost(4.0);
        let join = VisPlanNode::binary(VisOperator::HashJoin, "?s", scan1, scan2)
            .with_estimated_rows(600)
            .with_cost(15.0);
        let filter = VisPlanNode::unary(VisOperator::Filter, "?age > 18", join)
            .with_estimated_rows(300)
            .with_cost(1.0);
        let sort = VisPlanNode::unary(VisOperator::Sort, "ORDER BY ?name", filter)
            .with_estimated_rows(300)
            .with_cost(8.0);
        VisPlanNode::unary(VisOperator::Projection, "?name ?age", sort)
            .with_estimated_rows(300)
            .with_cost(0.5)
    }

    // -- VisOperator tests --------------------------------------------------

    #[test]
    fn test_operator_display() {
        assert_eq!(VisOperator::Scan.to_string(), "Scan");
        assert_eq!(VisOperator::HashJoin.to_string(), "HashJoin");
        assert_eq!(VisOperator::Custom("MyOp".into()).to_string(), "MyOp");
    }

    #[test]
    fn test_operator_short_label() {
        assert_eq!(VisOperator::Scan.short_label(), "Scan");
        assert_eq!(VisOperator::HashJoin.short_label(), "HJ");
        assert_eq!(VisOperator::Projection.short_label(), "Pi");
    }

    #[test]
    fn test_operator_is_join() {
        assert!(VisOperator::HashJoin.is_join());
        assert!(VisOperator::MergeJoin.is_join());
        assert!(VisOperator::NestedLoopJoin.is_join());
        assert!(VisOperator::LeftJoin.is_join());
        assert!(VisOperator::LateralJoin.is_join());
        assert!(!VisOperator::Scan.is_join());
        assert!(!VisOperator::Filter.is_join());
        assert!(!VisOperator::Projection.is_join());
    }

    // -- VisPlanNode builder tests ------------------------------------------

    #[test]
    fn test_leaf_node() {
        let node = simple_scan();
        assert_eq!(node.operator, VisOperator::Scan);
        assert_eq!(node.description, "?s ?p ?o");
        assert_eq!(node.estimated_rows, Some(1000));
        assert_eq!(node.estimated_cost, Some(5.0));
        assert!(node.children.is_empty());
    }

    #[test]
    fn test_unary_node() {
        let child = simple_scan();
        let node = VisPlanNode::unary(VisOperator::Filter, "?x > 5", child);
        assert_eq!(node.children.len(), 1);
        assert_eq!(node.operator, VisOperator::Filter);
    }

    #[test]
    fn test_binary_node() {
        let node = two_scan_join();
        assert_eq!(node.children.len(), 2);
        assert_eq!(node.operator, VisOperator::HashJoin);
    }

    #[test]
    fn test_with_property() {
        let node = simple_scan().with_property("index", "spo");
        assert_eq!(
            node.properties.get("index").map(|s| s.as_str()),
            Some("spo")
        );
    }

    #[test]
    fn test_with_actual_rows() {
        let node = simple_scan().with_actual_rows(950);
        assert_eq!(node.actual_rows, Some(950));
    }

    #[test]
    fn test_with_execution_time() {
        let node = simple_scan().with_execution_time_us(1234);
        assert_eq!(node.execution_time_us, Some(1234));
    }

    // -- tree metrics -------------------------------------------------------

    #[test]
    fn test_node_count_leaf() {
        assert_eq!(simple_scan().node_count(), 1);
    }

    #[test]
    fn test_node_count_complex() {
        // Projection -> Sort -> Filter -> HashJoin(Scan, Scan) = 6
        assert_eq!(complex_plan().node_count(), 6);
    }

    #[test]
    fn test_depth_leaf() {
        assert_eq!(simple_scan().depth(), 1);
    }

    #[test]
    fn test_depth_complex() {
        // depth: Projection -> Sort -> Filter -> HashJoin -> Scan = 5
        assert_eq!(complex_plan().depth(), 5);
    }

    #[test]
    fn test_collect_filters() {
        let plan = complex_plan();
        let filters = plan.collect_filters();
        assert_eq!(filters.len(), 1);
        assert_eq!(filters[0].description, "?age > 18");
    }

    #[test]
    fn test_collect_joins() {
        let plan = complex_plan();
        let joins = plan.collect_joins();
        assert_eq!(joins.len(), 1);
        assert_eq!(joins[0].operator, VisOperator::HashJoin);
    }

    #[test]
    fn test_total_cost() {
        let plan = complex_plan();
        // 5.0 + 4.0 + 15.0 + 1.0 + 8.0 + 0.5 = 33.5
        let cost = plan.total_cost();
        assert!((cost - 33.5).abs() < 1e-6);
    }

    // -- DOT output ---------------------------------------------------------

    #[test]
    fn test_dot_output_contains_digraph() {
        let viz = QueryPlanVisualizer::new();
        let dot = viz.render(&simple_scan(), VisOutputFormat::Dot);
        assert!(dot.is_ok());
        let dot = dot.unwrap_or_default();
        assert!(dot.contains("digraph QueryPlan"));
        assert!(dot.contains("rankdir=TB"));
    }

    #[test]
    fn test_dot_output_has_nodes() {
        let viz = QueryPlanVisualizer::new();
        let dot = viz.render(&two_scan_join(), VisOutputFormat::Dot);
        let dot = dot.unwrap_or_default();
        assert!(dot.contains("n1 "));
        assert!(dot.contains("n2 "));
        assert!(dot.contains("n3 "));
    }

    #[test]
    fn test_dot_output_has_edges() {
        let viz = QueryPlanVisualizer::new();
        let dot = viz.render(&two_scan_join(), VisOutputFormat::Dot);
        let dot = dot.unwrap_or_default();
        assert!(dot.contains("n1 -> n2"));
        assert!(dot.contains("n1 -> n3"));
        assert!(dot.contains("left"));
        assert!(dot.contains("right"));
    }

    #[test]
    fn test_dot_complex_plan() {
        let viz = QueryPlanVisualizer::new();
        let result = viz.render(&complex_plan(), VisOutputFormat::Dot);
        assert!(result.is_ok());
        let dot = result.unwrap_or_default();
        assert!(dot.contains("HJ"));
        assert!(dot.contains("Sort"));
    }

    #[test]
    fn test_dot_orientation_lr() {
        let config = VisualizerConfig {
            dot_orientation: "LR".to_string(),
            ..Default::default()
        };
        let viz = QueryPlanVisualizer::with_config(config);
        let dot = viz
            .render(&simple_scan(), VisOutputFormat::Dot)
            .unwrap_or_default();
        assert!(dot.contains("rankdir=LR"));
    }

    #[test]
    fn test_dot_no_colour() {
        let config = VisualizerConfig {
            use_colour: false,
            ..Default::default()
        };
        let viz = QueryPlanVisualizer::with_config(config);
        let dot = viz
            .render(&simple_scan(), VisOutputFormat::Dot)
            .unwrap_or_default();
        assert!(dot.contains("white"));
    }

    #[test]
    fn test_dot_escape_special_chars() {
        let node = VisPlanNode::leaf(VisOperator::Filter, "?x < 10 && ?y > 5");
        let viz = QueryPlanVisualizer::new();
        let dot = viz.render(&node, VisOutputFormat::Dot).unwrap_or_default();
        // < and > should be escaped
        assert!(dot.contains("\\<") || dot.contains("\\>"));
    }

    // -- Text tree output ---------------------------------------------------

    #[test]
    fn test_text_tree_simple() {
        let viz = QueryPlanVisualizer::new();
        let tree = viz
            .render(&simple_scan(), VisOutputFormat::TextTree)
            .unwrap_or_default();
        assert!(tree.contains("[Scan]"));
        assert!(tree.contains("?s ?p ?o"));
    }

    #[test]
    fn test_text_tree_shows_estimates() {
        let viz = QueryPlanVisualizer::new();
        let tree = viz
            .render(&simple_scan(), VisOutputFormat::TextTree)
            .unwrap_or_default();
        assert!(tree.contains("est=1000"));
        assert!(tree.contains("cost=5.0"));
    }

    #[test]
    fn test_text_tree_join() {
        let viz = QueryPlanVisualizer::new();
        let tree = viz
            .render(&two_scan_join(), VisOutputFormat::TextTree)
            .unwrap_or_default();
        assert!(tree.contains("[HashJoin]"));
        assert!(tree.contains(":name"));
        assert!(tree.contains(":age"));
    }

    #[test]
    fn test_text_tree_complex() {
        let viz = QueryPlanVisualizer::new();
        let tree = viz
            .render(&complex_plan(), VisOutputFormat::TextTree)
            .unwrap_or_default();
        assert!(tree.contains("[Projection]"));
        assert!(tree.contains("[Sort]"));
        assert!(tree.contains("[Filter]"));
        assert!(tree.contains("[HashJoin]"));
        assert!(tree.contains("[Scan]"));
    }

    #[test]
    fn test_text_tree_with_properties() {
        let node =
            VisPlanNode::leaf(VisOperator::Scan, "?s ?p ?o").with_property("index_used", "spo");
        let viz = QueryPlanVisualizer::new();
        let tree = viz
            .render(&node, VisOutputFormat::TextTree)
            .unwrap_or_default();
        assert!(tree.contains("index_used: spo"));
    }

    #[test]
    fn test_text_tree_no_estimates() {
        let config = VisualizerConfig {
            show_estimated_rows: false,
            show_cost: false,
            ..Default::default()
        };
        let viz = QueryPlanVisualizer::with_config(config);
        let tree = viz
            .render(&simple_scan(), VisOutputFormat::TextTree)
            .unwrap_or_default();
        assert!(!tree.contains("est="));
        assert!(!tree.contains("cost="));
    }

    #[test]
    fn test_text_tree_actual_rows_and_time() {
        let node = simple_scan()
            .with_actual_rows(950)
            .with_execution_time_us(1234);
        let viz = QueryPlanVisualizer::new();
        let tree = viz
            .render(&node, VisOutputFormat::TextTree)
            .unwrap_or_default();
        assert!(tree.contains("actual=950"));
        assert!(tree.contains("time=1234us"));
    }

    // -- JSON output --------------------------------------------------------

    #[test]
    fn test_json_output_parses() {
        let viz = QueryPlanVisualizer::new();
        let json_str = viz
            .render(&simple_scan(), VisOutputFormat::Json)
            .unwrap_or_default();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("valid JSON");
        assert!(parsed.is_object());
    }

    #[test]
    fn test_json_contains_operator() {
        let viz = QueryPlanVisualizer::new();
        let json_str = viz
            .render(&simple_scan(), VisOutputFormat::Json)
            .unwrap_or_default();
        assert!(json_str.contains("\"Scan\""));
    }

    #[test]
    fn test_json_roundtrip() {
        let viz = QueryPlanVisualizer::new();
        let original = complex_plan();
        let json_str = viz
            .render(&original, VisOutputFormat::Json)
            .unwrap_or_default();
        let deserialized: VisPlanNode = serde_json::from_str(&json_str).expect("deserialise");
        assert_eq!(deserialized.operator, VisOperator::Projection);
        assert_eq!(deserialized.children.len(), 1);
    }

    #[test]
    fn test_json_compact() {
        let config = VisualizerConfig {
            json_pretty: false,
            ..Default::default()
        };
        let viz = QueryPlanVisualizer::with_config(config);
        let json_str = viz
            .render(&simple_scan(), VisOutputFormat::Json)
            .unwrap_or_default();
        // Compact JSON should not contain leading newlines inside object
        assert!(!json_str.contains("\n  "));
    }

    // -- Summary statistics -------------------------------------------------

    #[test]
    fn test_summary_simple() {
        let viz = QueryPlanVisualizer::new();
        let summary = viz.summarise(&simple_scan());
        assert_eq!(summary.total_nodes, 1);
        assert_eq!(summary.max_depth, 1);
        assert_eq!(summary.scan_count, 1);
        assert_eq!(summary.join_count, 0);
        assert_eq!(summary.filter_count, 0);
    }

    #[test]
    fn test_summary_complex() {
        let viz = QueryPlanVisualizer::new();
        let summary = viz.summarise(&complex_plan());
        assert_eq!(summary.total_nodes, 6);
        assert_eq!(summary.join_count, 1);
        assert_eq!(summary.filter_count, 1);
        assert_eq!(summary.scan_count, 2);
        assert!((summary.total_estimated_cost - 33.5).abs() < 1e-6);
    }

    #[test]
    fn test_summary_operator_histogram() {
        let viz = QueryPlanVisualizer::new();
        let summary = viz.summarise(&complex_plan());
        assert_eq!(summary.operator_histogram.get("Scan"), Some(&2));
        assert_eq!(summary.operator_histogram.get("HashJoin"), Some(&1));
        assert_eq!(summary.operator_histogram.get("Filter"), Some(&1));
    }

    // -- VisOutputFormat display -------------------------------------------

    #[test]
    fn test_output_format_display() {
        assert_eq!(VisOutputFormat::Dot.to_string(), "DOT");
        assert_eq!(VisOutputFormat::TextTree.to_string(), "TextTree");
        assert_eq!(VisOutputFormat::Json.to_string(), "JSON");
    }

    // -- Config defaults ---------------------------------------------------

    #[test]
    fn test_default_config() {
        let config = VisualizerConfig::default();
        assert!(config.show_estimated_rows);
        assert!(config.show_cost);
        assert!(config.use_colour);
        assert_eq!(config.dot_orientation, "TB");
        assert!(config.json_pretty);
    }

    // -- Edge cases ---------------------------------------------------------

    #[test]
    fn test_empty_description() {
        let node = VisPlanNode::leaf(VisOperator::Scan, "");
        let viz = QueryPlanVisualizer::new();
        let dot = viz.render(&node, VisOutputFormat::Dot);
        assert!(dot.is_ok());
    }

    #[test]
    fn test_deeply_nested_plan() {
        let mut current = VisPlanNode::leaf(VisOperator::Scan, "?s ?p ?o");
        for i in 0..10 {
            current = VisPlanNode::unary(VisOperator::Filter, format!("filter_{i}"), current);
        }
        let viz = QueryPlanVisualizer::new();
        let result = viz.render(&current, VisOutputFormat::TextTree);
        assert!(result.is_ok());
        assert_eq!(current.depth(), 11);
    }

    #[test]
    fn test_union_three_branches() {
        let s1 = VisPlanNode::leaf(VisOperator::Scan, "branch1");
        let s2 = VisPlanNode::leaf(VisOperator::Scan, "branch2");
        let s3 = VisPlanNode::leaf(VisOperator::Scan, "branch3");
        let union = VisPlanNode {
            id: 0,
            operator: VisOperator::Union,
            description: "UNION".into(),
            estimated_rows: None,
            estimated_cost: None,
            actual_rows: None,
            execution_time_us: None,
            properties: HashMap::new(),
            children: vec![s1, s2, s3],
        };
        assert_eq!(union.node_count(), 4);
        let viz = QueryPlanVisualizer::new();
        let result = viz.render(&union, VisOutputFormat::Dot);
        assert!(result.is_ok());
    }

    #[test]
    fn test_service_node() {
        let remote = VisPlanNode::leaf(VisOperator::Scan, "?s ?p ?o")
            .with_property("endpoint", "http://dbpedia.org/sparql");
        let svc = VisPlanNode::unary(
            VisOperator::Service,
            "SERVICE <http://dbpedia.org/sparql>",
            remote,
        );
        let viz = QueryPlanVisualizer::new();
        let tree = viz
            .render(&svc, VisOutputFormat::TextTree)
            .unwrap_or_default();
        assert!(tree.contains("[Service]"));
    }

    #[test]
    fn test_total_cost_no_costs() {
        let node = VisPlanNode::leaf(VisOperator::Scan, "?s ?p ?o");
        assert!((node.total_cost() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_custom_operator() {
        let node = VisPlanNode::leaf(VisOperator::Custom("GeoFilter".into()), "WITHIN polygon");
        assert_eq!(node.operator.to_string(), "GeoFilter");
        assert!(!node.operator.is_join());
    }

    #[test]
    fn test_all_formats_produce_output() {
        let plan = complex_plan();
        let viz = QueryPlanVisualizer::new();
        for fmt in [
            VisOutputFormat::Dot,
            VisOutputFormat::TextTree,
            VisOutputFormat::Json,
        ] {
            let result = viz.render(&plan, fmt);
            assert!(result.is_ok(), "format {fmt} should succeed");
            assert!(!result.unwrap_or_default().is_empty());
        }
    }

    #[test]
    fn test_left_join_node() {
        let left = VisPlanNode::leaf(VisOperator::Scan, "?s :name ?name");
        let right = VisPlanNode::leaf(VisOperator::Scan, "?s :email ?email");
        let optional = VisPlanNode::binary(VisOperator::LeftJoin, "OPTIONAL", left, right);
        assert!(optional.operator.is_join());
        let viz = QueryPlanVisualizer::new();
        let tree = viz
            .render(&optional, VisOutputFormat::TextTree)
            .unwrap_or_default();
        assert!(tree.contains("[LeftJoin]"));
    }

    #[test]
    fn test_values_node() {
        let vals = VisPlanNode::leaf(VisOperator::Values, "VALUES (?x) { (1) (2) (3) }");
        let viz = QueryPlanVisualizer::new();
        let dot = viz.render(&vals, VisOutputFormat::Dot).unwrap_or_default();
        assert!(dot.contains("Val"));
    }

    #[test]
    fn test_slice_node() {
        let scan = simple_scan();
        let slice = VisPlanNode::unary(VisOperator::Slice, "LIMIT 10 OFFSET 5", scan)
            .with_estimated_rows(10);
        let viz = QueryPlanVisualizer::new();
        let tree = viz
            .render(&slice, VisOutputFormat::TextTree)
            .unwrap_or_default();
        assert!(tree.contains("[Slice]"));
        assert!(tree.contains("LIMIT 10 OFFSET 5"));
    }

    #[test]
    fn test_aggregate_node() {
        let scan = simple_scan();
        let agg = VisPlanNode::unary(VisOperator::Aggregate, "GROUP BY ?type; COUNT(?s)", scan)
            .with_estimated_rows(50);
        assert_eq!(agg.depth(), 2);
    }

    #[test]
    fn test_materialise_node() {
        let scan = simple_scan();
        let mat = VisPlanNode::unary(VisOperator::Materialise, "materialised", scan);
        let viz = QueryPlanVisualizer::new();
        let dot = viz.render(&mat, VisOutputFormat::Dot).unwrap_or_default();
        assert!(dot.contains("Mat"));
    }

    #[test]
    fn test_graph_node() {
        let scan = simple_scan();
        let graph = VisPlanNode::unary(VisOperator::Graph, "GRAPH <http://example.org/g1>", scan);
        let viz = QueryPlanVisualizer::new();
        let tree = viz
            .render(&graph, VisOutputFormat::TextTree)
            .unwrap_or_default();
        assert!(tree.contains("[Graph]"));
    }

    #[test]
    fn test_distinct_node() {
        let scan = simple_scan();
        let distinct = VisPlanNode::unary(VisOperator::Distinct, "DISTINCT", scan);
        let viz = QueryPlanVisualizer::new();
        let dot = viz
            .render(&distinct, VisOutputFormat::Dot)
            .unwrap_or_default();
        assert!(dot.contains("Dist"));
    }

    #[test]
    fn test_bind_node() {
        let scan = simple_scan();
        let bind = VisPlanNode::unary(VisOperator::Bind, "BIND(?x + 1 AS ?y)", scan);
        assert_eq!(bind.node_count(), 2);
    }

    #[test]
    fn test_subquery_node() {
        let inner = VisPlanNode::leaf(VisOperator::Scan, "inner pattern");
        let sq = VisPlanNode::unary(VisOperator::SubQuery, "sub-select", inner);
        assert_eq!(sq.depth(), 2);
    }

    #[test]
    fn test_multiple_properties() {
        let node = VisPlanNode::leaf(VisOperator::Scan, "?s ?p ?o")
            .with_property("index", "spo")
            .with_property("selectivity", "0.01");
        assert_eq!(node.properties.len(), 2);
    }
}
