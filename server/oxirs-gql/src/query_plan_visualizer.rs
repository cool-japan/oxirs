//! Query Plan Visualization
//!
//! This module provides visualization and analysis tools for GraphQL query execution plans.
//! It helps developers understand how queries are executed, identify bottlenecks, and optimize performance.
//!
//! # Features
//!
//! - **Visual Query Plans**: Generate visual representations of execution plans
//! - **Multiple Formats**: DOT (Graphviz), Mermaid, ASCII tree, JSON
//! - **Execution Timeline**: Visualize execution order and timing
//! - **Cost Analysis**: Display estimated costs for each operation
//! - **Dependency Graphs**: Show field dependencies and parallelization opportunities
//! - **Performance Metrics**: Integration with actual execution metrics
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_gql::query_plan_visualizer::{QueryPlanVisualizer, VisualizationFormat};
//!
//! let visualizer = QueryPlanVisualizer::new();
//! let plan = /* ... execution plan ... */;
//!
//! // Generate Graphviz DOT format
//! let dot = visualizer.visualize(&plan, VisualizationFormat::Dot)?;
//!
//! // Generate Mermaid diagram
//! let mermaid = visualizer.visualize(&plan, VisualizationFormat::Mermaid)?;
//!
//! // Generate ASCII tree
//! let tree = visualizer.visualize(&plan, VisualizationFormat::AsciiTree)?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

/// Visualization format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualizationFormat {
    /// Graphviz DOT format
    Dot,
    /// Mermaid diagram
    Mermaid,
    /// ASCII tree
    AsciiTree,
    /// JSON representation
    Json,
    /// HTML with interactive visualization
    Html,
}

/// Query plan node representing a single operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanNode {
    /// Unique node ID
    pub id: String,
    /// Node type (e.g., "Field", "Object", "List", "Defer", "Stream")
    pub node_type: String,
    /// Display name
    pub name: String,
    /// Parent node ID
    pub parent_id: Option<String>,
    /// Child node IDs
    pub children: Vec<String>,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Actual execution time (if available)
    pub execution_time_ms: Option<u64>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Whether this can be executed in parallel with siblings
    pub parallel_execution: bool,
    /// Dependencies (node IDs that must complete first)
    pub dependencies: Vec<String>,
}

impl PlanNode {
    /// Create a new plan node
    pub fn new(id: String, node_type: String, name: String) -> Self {
        Self {
            id,
            node_type,
            name,
            parent_id: None,
            children: Vec::new(),
            estimated_cost: 1.0,
            execution_time_ms: None,
            metadata: HashMap::new(),
            parallel_execution: true,
            dependencies: Vec::new(),
        }
    }

    /// Set parent
    pub fn with_parent(mut self, parent_id: String) -> Self {
        self.parent_id = Some(parent_id);
        self
    }

    /// Add child
    pub fn with_child(mut self, child_id: String) -> Self {
        self.children.push(child_id);
        self
    }

    /// Set estimated cost
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.estimated_cost = cost;
        self
    }

    /// Set execution time
    pub fn with_execution_time(mut self, time_ms: u64) -> Self {
        self.execution_time_ms = Some(time_ms);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set parallel execution capability
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel_execution = parallel;
        self
    }

    /// Add dependency
    pub fn with_dependency(mut self, dep_id: String) -> Self {
        self.dependencies.push(dep_id);
        self
    }
}

/// Query execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    /// Root node ID
    pub root_id: String,
    /// All nodes in the plan
    pub nodes: HashMap<String, PlanNode>,
    /// Query text
    pub query: String,
    /// Total estimated cost
    pub total_cost: f64,
    /// Total execution time (if available)
    pub total_execution_time_ms: Option<u64>,
}

impl QueryPlan {
    /// Create a new query plan
    pub fn new(root_id: String, query: String) -> Self {
        Self {
            root_id,
            nodes: HashMap::new(),
            query,
            total_cost: 0.0,
            total_execution_time_ms: None,
        }
    }

    /// Add a node to the plan
    pub fn add_node(&mut self, node: PlanNode) {
        self.total_cost += node.estimated_cost;
        self.nodes.insert(node.id.clone(), node);
    }

    /// Get root node
    pub fn root_node(&self) -> Option<&PlanNode> {
        self.nodes.get(&self.root_id)
    }

    /// Get node by ID
    pub fn get_node(&self, id: &str) -> Option<&PlanNode> {
        self.nodes.get(id)
    }

    /// Get all leaf nodes (nodes with no children)
    pub fn leaf_nodes(&self) -> Vec<&PlanNode> {
        self.nodes
            .values()
            .filter(|node| node.children.is_empty())
            .collect()
    }

    /// Calculate execution levels (depth from root)
    pub fn execution_levels(&self) -> HashMap<String, usize> {
        let mut levels = HashMap::new();
        let mut queue = vec![(self.root_id.clone(), 0)];

        while let Some((node_id, level)) = queue.pop() {
            levels.insert(node_id.clone(), level);

            if let Some(node) = self.nodes.get(&node_id) {
                for child_id in &node.children {
                    queue.push((child_id.clone(), level + 1));
                }
            }
        }

        levels
    }

    /// Get nodes that can be executed in parallel at each level
    pub fn parallel_execution_groups(&self) -> Vec<Vec<String>> {
        let levels = self.execution_levels();
        let max_level = levels.values().max().copied().unwrap_or(0);

        let mut groups = vec![Vec::new(); max_level + 1];

        for (node_id, level) in levels {
            if let Some(node) = self.nodes.get(&node_id) {
                if node.parallel_execution && node.dependencies.is_empty() {
                    groups[level].push(node_id);
                }
            }
        }

        groups
    }

    /// Set total execution time
    pub fn set_total_execution_time(&mut self, time_ms: u64) {
        self.total_execution_time_ms = Some(time_ms);
    }
}

/// Query plan visualizer
pub struct QueryPlanVisualizer {
    /// Show costs in visualization
    pub show_costs: bool,
    /// Show execution times in visualization
    pub show_execution_times: bool,
    /// Show metadata in visualization
    pub show_metadata: bool,
    /// Color scheme for different node types
    pub colors: HashMap<String, String>,
}

impl QueryPlanVisualizer {
    /// Create a new visualizer with default settings
    pub fn new() -> Self {
        let mut colors = HashMap::new();
        colors.insert("Field".to_string(), "#87CEEB".to_string()); // Sky blue
        colors.insert("Object".to_string(), "#90EE90".to_string()); // Light green
        colors.insert("List".to_string(), "#FFB6C1".to_string()); // Light pink
        colors.insert("Defer".to_string(), "#FFD700".to_string()); // Gold
        colors.insert("Stream".to_string(), "#FF6347".to_string()); // Tomato

        Self {
            show_costs: true,
            show_execution_times: true,
            show_metadata: false,
            colors,
        }
    }

    /// Enable cost display
    pub fn with_costs(mut self, show: bool) -> Self {
        self.show_costs = show;
        self
    }

    /// Enable execution time display
    pub fn with_execution_times(mut self, show: bool) -> Self {
        self.show_execution_times = show;
        self
    }

    /// Enable metadata display
    pub fn with_metadata(mut self, show: bool) -> Self {
        self.show_metadata = show;
        self
    }

    /// Visualize a query plan
    pub fn visualize(
        &self,
        plan: &QueryPlan,
        format: VisualizationFormat,
    ) -> Result<String, VisualizationError> {
        match format {
            VisualizationFormat::Dot => self.visualize_dot(plan),
            VisualizationFormat::Mermaid => self.visualize_mermaid(plan),
            VisualizationFormat::AsciiTree => self.visualize_ascii_tree(plan),
            VisualizationFormat::Json => self.visualize_json(plan),
            VisualizationFormat::Html => self.visualize_html(plan),
        }
    }

    /// Generate Graphviz DOT format
    fn visualize_dot(&self, plan: &QueryPlan) -> Result<String, VisualizationError> {
        let mut output = String::new();
        writeln!(output, "digraph QueryPlan {{")
            .map_err(|e| VisualizationError::FormatError(e.to_string()))?;
        writeln!(output, "  rankdir=TB;")
            .map_err(|e| VisualizationError::FormatError(e.to_string()))?;
        writeln!(output, "  node [shape=box, style=filled];")
            .map_err(|e| VisualizationError::FormatError(e.to_string()))?;

        // Add nodes
        for (id, node) in &plan.nodes {
            let default_color = "#FFFFFF".to_string();
            let color = self.colors.get(&node.node_type).unwrap_or(&default_color);

            let mut label = node.name.clone();
            if self.show_costs {
                label.push_str(&format!("\\nCost: {:.2}", node.estimated_cost));
            }
            if self.show_execution_times {
                if let Some(time) = node.execution_time_ms {
                    label.push_str(&format!("\\nTime: {}ms", time));
                }
            }

            writeln!(
                output,
                "  \"{}\" [label=\"{}\", fillcolor=\"{}\"];",
                id, label, color
            )
            .map_err(|e| VisualizationError::FormatError(e.to_string()))?;
        }

        // Add edges
        for node in plan.nodes.values() {
            for child_id in &node.children {
                writeln!(output, "  \"{}\" -> \"{}\";", node.id, child_id)
                    .map_err(|e| VisualizationError::FormatError(e.to_string()))?;
            }

            // Add dependency edges (dashed)
            for dep_id in &node.dependencies {
                writeln!(
                    output,
                    "  \"{}\" -> \"{}\" [style=dashed, color=red];",
                    dep_id, node.id
                )
                .map_err(|e| VisualizationError::FormatError(e.to_string()))?;
            }
        }

        writeln!(output, "}}").map_err(|e| VisualizationError::FormatError(e.to_string()))?;
        Ok(output)
    }

    /// Generate Mermaid diagram
    fn visualize_mermaid(&self, plan: &QueryPlan) -> Result<String, VisualizationError> {
        let mut output = String::new();
        writeln!(output, "graph TD").map_err(|e| VisualizationError::FormatError(e.to_string()))?;

        // Add nodes
        for (id, node) in &plan.nodes {
            let mut label = node.name.clone();
            if self.show_costs {
                label.push_str(&format!("<br/>Cost: {:.2}", node.estimated_cost));
            }
            if self.show_execution_times {
                if let Some(time) = node.execution_time_ms {
                    label.push_str(&format!("<br/>Time: {}ms", time));
                }
            }

            writeln!(output, "  {}[\"{}\"]", id.replace('-', "_"), label)
                .map_err(|e| VisualizationError::FormatError(e.to_string()))?;
        }

        // Add edges
        for node in plan.nodes.values() {
            for child_id in &node.children {
                writeln!(
                    output,
                    "  {} --> {}",
                    node.id.replace('-', "_"),
                    child_id.replace('-', "_")
                )
                .map_err(|e| VisualizationError::FormatError(e.to_string()))?;
            }

            // Add dependency edges
            for dep_id in &node.dependencies {
                writeln!(
                    output,
                    "  {} -.-> {}",
                    dep_id.replace('-', "_"),
                    node.id.replace('-', "_")
                )
                .map_err(|e| VisualizationError::FormatError(e.to_string()))?;
            }
        }

        Ok(output)
    }

    /// Generate ASCII tree
    fn visualize_ascii_tree(&self, plan: &QueryPlan) -> Result<String, VisualizationError> {
        let mut output = String::new();
        writeln!(output, "Query Plan:")
            .map_err(|e| VisualizationError::FormatError(e.to_string()))?;
        writeln!(output, "Total Cost: {:.2}", plan.total_cost)
            .map_err(|e| VisualizationError::FormatError(e.to_string()))?;

        if let Some(time) = plan.total_execution_time_ms {
            writeln!(output, "Total Time: {}ms", time)
                .map_err(|e| VisualizationError::FormatError(e.to_string()))?;
        }

        writeln!(output).map_err(|e| VisualizationError::FormatError(e.to_string()))?;

        if let Some(root) = plan.root_node() {
            self.print_tree_node(plan, root, "", true, &mut output)?;
        }

        Ok(output)
    }

    fn print_tree_node(
        &self,
        plan: &QueryPlan,
        node: &PlanNode,
        prefix: &str,
        is_last: bool,
        output: &mut String,
    ) -> Result<(), VisualizationError> {
        let connector = if is_last { "└── " } else { "├── " };
        let mut line = format!("{}{}{}", prefix, connector, node.name);

        if self.show_costs {
            line.push_str(&format!(" (cost: {:.2}", node.estimated_cost));
            if self.show_execution_times {
                if let Some(time) = node.execution_time_ms {
                    line.push_str(&format!(", time: {}ms", time));
                }
            }
            line.push(')');
        } else if self.show_execution_times {
            if let Some(time) = node.execution_time_ms {
                line.push_str(&format!(" (time: {}ms)", time));
            }
        }

        writeln!(output, "{}", line).map_err(|e| VisualizationError::FormatError(e.to_string()))?;

        let child_prefix = format!("{}{}   ", prefix, if is_last { " " } else { "│" });

        for (i, child_id) in node.children.iter().enumerate() {
            if let Some(child) = plan.get_node(child_id) {
                let is_last_child = i == node.children.len() - 1;
                self.print_tree_node(plan, child, &child_prefix, is_last_child, output)?;
            }
        }

        Ok(())
    }

    /// Generate JSON representation
    fn visualize_json(&self, plan: &QueryPlan) -> Result<String, VisualizationError> {
        serde_json::to_string_pretty(plan)
            .map_err(|e| VisualizationError::FormatError(e.to_string()))
    }

    /// Generate HTML with interactive visualization
    fn visualize_html(&self, plan: &QueryPlan) -> Result<String, VisualizationError> {
        let json = self.visualize_json(plan)?;

        Ok(format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Query Plan Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .node {{ border: 1px solid #ccc; padding: 10px; margin: 5px; border-radius: 5px; }}
        .node-Field {{ background-color: #87CEEB; }}
        .node-Object {{ background-color: #90EE90; }}
        .node-List {{ background-color: #FFB6C1; }}
        .node-Defer {{ background-color: #FFD700; }}
        .node-Stream {{ background-color: #FF6347; }}
        .metadata {{ font-size: 0.9em; color: #666; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Query Plan Visualization</h1>
    <div id="visualization"></div>
    <h2>JSON Representation</h2>
    <pre>{}</pre>
    <script>
        const plan = {};
        // Interactive visualization would go here
        console.log('Query plan:', plan);
    </script>
</body>
</html>"#,
            json, json
        ))
    }

    /// Generate execution timeline
    pub fn generate_timeline(&self, plan: &QueryPlan) -> String {
        let mut output = String::new();
        writeln!(&mut output, "Execution Timeline:").expect("writing to String should not fail");
        writeln!(&mut output, "==================").expect("writing to String should not fail");

        let levels = plan.execution_levels();
        let max_level = levels.values().max().copied().unwrap_or(0);

        for level in 0..=max_level {
            let nodes_at_level: Vec<_> = plan
                .nodes
                .iter()
                .filter(|(id, _)| levels.get(*id).copied() == Some(level))
                .collect();

            if !nodes_at_level.is_empty() {
                writeln!(&mut output, "\nLevel {}:", level)
                    .expect("writing to String should not fail");
                for (_id, node) in nodes_at_level {
                    let time_str = node
                        .execution_time_ms
                        .map(|t| format!(" ({}ms)", t))
                        .unwrap_or_default();
                    let parallel = if node.parallel_execution {
                        " [parallel]"
                    } else {
                        ""
                    };
                    writeln!(
                        &mut output,
                        "  - {} [{}]{}{}",
                        node.name, node.node_type, time_str, parallel
                    )
                    .expect("writing to String should not fail");
                }
            }
        }

        output
    }

    /// Generate cost breakdown
    pub fn generate_cost_breakdown(&self, plan: &QueryPlan) -> String {
        let mut output = String::new();
        writeln!(&mut output, "Cost Breakdown:").expect("writing to String should not fail");
        writeln!(&mut output, "===============").expect("writing to String should not fail");
        writeln!(&mut output, "Total Cost: {:.2}\n", plan.total_cost)
            .expect("writing to String should not fail");

        let mut nodes: Vec<_> = plan.nodes.values().collect();
        nodes.sort_by(|a, b| {
            b.estimated_cost
                .partial_cmp(&a.estimated_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for node in nodes {
            let percentage = (node.estimated_cost / plan.total_cost) * 100.0;
            writeln!(
                &mut output,
                "{:<30} {:>8.2} ({:>5.1}%)",
                node.name, node.estimated_cost, percentage
            )
            .expect("writing to String should not fail");
        }

        output
    }
}

impl Default for QueryPlanVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during visualization
#[derive(Debug, thiserror::Error)]
pub enum VisualizationError {
    /// Format error
    #[error("Formatting error: {0}")]
    FormatError(String),

    /// Invalid plan structure
    #[error("Invalid plan structure: {0}")]
    InvalidPlan(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_sample_plan() -> QueryPlan {
        let mut plan = QueryPlan::new("root".to_string(), "{ user { posts } }".to_string());

        let root = PlanNode::new(
            "root".to_string(),
            "Object".to_string(),
            "Query".to_string(),
        )
        .with_cost(10.0)
        .with_child("user".to_string());

        let user = PlanNode::new("user".to_string(), "Field".to_string(), "user".to_string())
            .with_parent("root".to_string())
            .with_cost(5.0)
            .with_child("posts".to_string())
            .with_execution_time(15);

        let posts = PlanNode::new("posts".to_string(), "List".to_string(), "posts".to_string())
            .with_parent("user".to_string())
            .with_cost(20.0)
            .with_execution_time(45);

        plan.add_node(root);
        plan.add_node(user);
        plan.add_node(posts);
        plan.set_total_execution_time(60);

        plan
    }

    #[test]
    fn test_plan_node_creation() {
        let node = PlanNode::new(
            "test".to_string(),
            "Field".to_string(),
            "testField".to_string(),
        )
        .with_cost(5.0)
        .with_execution_time(10)
        .with_parallel(true)
        .with_metadata("key".to_string(), serde_json::json!("value"));

        assert_eq!(node.id, "test");
        assert_eq!(node.node_type, "Field");
        assert_eq!(node.name, "testField");
        assert_eq!(node.estimated_cost, 5.0);
        assert_eq!(node.execution_time_ms, Some(10));
        assert!(node.parallel_execution);
        assert_eq!(node.metadata.len(), 1);
    }

    #[test]
    fn test_query_plan_creation() {
        let plan = QueryPlan::new("root".to_string(), "{ test }".to_string());

        assert_eq!(plan.root_id, "root");
        assert_eq!(plan.query, "{ test }");
        assert_eq!(plan.total_cost, 0.0);
        assert!(plan.nodes.is_empty());
    }

    #[test]
    fn test_add_node_to_plan() {
        let mut plan = QueryPlan::new("root".to_string(), "{ test }".to_string());
        let node = PlanNode::new("test".to_string(), "Field".to_string(), "test".to_string())
            .with_cost(5.0);

        plan.add_node(node);

        assert_eq!(plan.nodes.len(), 1);
        assert_eq!(plan.total_cost, 5.0);
    }

    #[test]
    fn test_root_node() {
        let plan = create_sample_plan();
        let root = plan.root_node();

        assert!(root.is_some());
        assert_eq!(root.unwrap().id, "root");
    }

    #[test]
    fn test_leaf_nodes() {
        let plan = create_sample_plan();
        let leaves = plan.leaf_nodes();

        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].id, "posts");
    }

    #[test]
    fn test_execution_levels() {
        let plan = create_sample_plan();
        let levels = plan.execution_levels();

        assert_eq!(levels.get("root"), Some(&0));
        assert_eq!(levels.get("user"), Some(&1));
        assert_eq!(levels.get("posts"), Some(&2));
    }

    #[test]
    fn test_parallel_execution_groups() {
        let mut plan = QueryPlan::new("root".to_string(), "{ test }".to_string());

        let root = PlanNode::new(
            "root".to_string(),
            "Object".to_string(),
            "Query".to_string(),
        )
        .with_child("field1".to_string())
        .with_child("field2".to_string());

        let field1 = PlanNode::new(
            "field1".to_string(),
            "Field".to_string(),
            "field1".to_string(),
        )
        .with_parent("root".to_string())
        .with_parallel(true);

        let field2 = PlanNode::new(
            "field2".to_string(),
            "Field".to_string(),
            "field2".to_string(),
        )
        .with_parent("root".to_string())
        .with_parallel(true);

        plan.add_node(root);
        plan.add_node(field1);
        plan.add_node(field2);

        let groups = plan.parallel_execution_groups();
        assert!(groups[1].len() == 2); // Both fields at level 1 can execute in parallel
    }

    #[test]
    fn test_visualizer_creation() {
        let visualizer = QueryPlanVisualizer::new();

        assert!(visualizer.show_costs);
        assert!(visualizer.show_execution_times);
        assert!(!visualizer.show_metadata);
        assert!(!visualizer.colors.is_empty());
    }

    #[test]
    fn test_visualizer_configuration() {
        let visualizer = QueryPlanVisualizer::new()
            .with_costs(false)
            .with_execution_times(false)
            .with_metadata(true);

        assert!(!visualizer.show_costs);
        assert!(!visualizer.show_execution_times);
        assert!(visualizer.show_metadata);
    }

    #[test]
    fn test_dot_visualization() {
        let plan = create_sample_plan();
        let visualizer = QueryPlanVisualizer::new();

        let dot = visualizer.visualize(&plan, VisualizationFormat::Dot);
        assert!(dot.is_ok());

        let output = dot.unwrap();
        assert!(output.contains("digraph QueryPlan"));
        assert!(output.contains("root"));
        assert!(output.contains("user"));
        assert!(output.contains("posts"));
    }

    #[test]
    fn test_mermaid_visualization() {
        let plan = create_sample_plan();
        let visualizer = QueryPlanVisualizer::new();

        let mermaid = visualizer.visualize(&plan, VisualizationFormat::Mermaid);
        assert!(mermaid.is_ok());

        let output = mermaid.unwrap();
        assert!(output.contains("graph TD"));
        assert!(output.contains("root"));
        assert!(output.contains("user"));
    }

    #[test]
    fn test_ascii_tree_visualization() {
        let plan = create_sample_plan();
        let visualizer = QueryPlanVisualizer::new();

        let tree = visualizer.visualize(&plan, VisualizationFormat::AsciiTree);
        assert!(tree.is_ok());

        let output = tree.unwrap();
        assert!(output.contains("Query Plan:"));
        assert!(output.contains("Total Cost:"));
        assert!(output.contains("Query"));
        assert!(output.contains("user"));
        assert!(output.contains("posts"));
    }

    #[test]
    fn test_json_visualization() {
        let plan = create_sample_plan();
        let visualizer = QueryPlanVisualizer::new();

        let json = visualizer.visualize(&plan, VisualizationFormat::Json);
        assert!(json.is_ok());

        let output = json.unwrap();
        assert!(output.contains("root_id"));
        assert!(output.contains("nodes"));
        assert!(output.contains("total_cost"));
    }

    #[test]
    fn test_html_visualization() {
        let plan = create_sample_plan();
        let visualizer = QueryPlanVisualizer::new();

        let html = visualizer.visualize(&plan, VisualizationFormat::Html);
        assert!(html.is_ok());

        let output = html.unwrap();
        assert!(output.contains("<!DOCTYPE html>"));
        assert!(output.contains("Query Plan Visualization"));
        assert!(output.contains("<script>"));
    }

    #[test]
    fn test_timeline_generation() {
        let plan = create_sample_plan();
        let visualizer = QueryPlanVisualizer::new();

        let timeline = visualizer.generate_timeline(&plan);
        assert!(timeline.contains("Execution Timeline"));
        assert!(timeline.contains("Level 0"));
        assert!(timeline.contains("Level 1"));
        assert!(timeline.contains("Level 2"));
    }

    #[test]
    fn test_cost_breakdown() {
        let plan = create_sample_plan();
        let visualizer = QueryPlanVisualizer::new();

        let breakdown = visualizer.generate_cost_breakdown(&plan);
        assert!(breakdown.contains("Cost Breakdown"));
        assert!(breakdown.contains("Total Cost:"));
        assert!(breakdown.contains("posts")); // Highest cost item
    }

    #[test]
    fn test_node_with_dependencies() {
        let node = PlanNode::new("test".to_string(), "Field".to_string(), "test".to_string())
            .with_dependency("dep1".to_string())
            .with_dependency("dep2".to_string());

        assert_eq!(node.dependencies.len(), 2);
    }

    #[test]
    fn test_plan_with_dependencies() {
        let mut plan = QueryPlan::new("root".to_string(), "{ test }".to_string());

        let node1 = PlanNode::new(
            "node1".to_string(),
            "Field".to_string(),
            "field1".to_string(),
        );
        let node2 = PlanNode::new(
            "node2".to_string(),
            "Field".to_string(),
            "field2".to_string(),
        )
        .with_dependency("node1".to_string());

        plan.add_node(node1);
        plan.add_node(node2);

        let node2_ref = plan.get_node("node2").unwrap();
        assert_eq!(node2_ref.dependencies.len(), 1);
        assert_eq!(node2_ref.dependencies[0], "node1");
    }

    #[test]
    fn test_dot_with_dependencies() {
        let mut plan = QueryPlan::new("root".to_string(), "{ test }".to_string());

        let node1 = PlanNode::new(
            "node1".to_string(),
            "Field".to_string(),
            "field1".to_string(),
        )
        .with_cost(1.0);
        let node2 = PlanNode::new(
            "node2".to_string(),
            "Field".to_string(),
            "field2".to_string(),
        )
        .with_dependency("node1".to_string())
        .with_cost(2.0);

        plan.add_node(node1);
        plan.add_node(node2);

        let visualizer = QueryPlanVisualizer::new();
        let dot = visualizer
            .visualize(&plan, VisualizationFormat::Dot)
            .unwrap();

        assert!(dot.contains("style=dashed"));
        assert!(dot.contains("color=red"));
    }

    #[test]
    fn test_complex_plan() {
        let mut plan = QueryPlan::new("root".to_string(), "complex query".to_string());

        // Build a more complex plan with multiple levels
        let root = PlanNode::new(
            "root".to_string(),
            "Object".to_string(),
            "Query".to_string(),
        )
        .with_child("user".to_string())
        .with_child("posts".to_string());

        let user = PlanNode::new("user".to_string(), "Object".to_string(), "User".to_string())
            .with_cost(5.0)
            .with_child("name".to_string())
            .with_child("email".to_string());

        let name = PlanNode::new("name".to_string(), "Field".to_string(), "name".to_string())
            .with_cost(1.0);

        let email = PlanNode::new(
            "email".to_string(),
            "Field".to_string(),
            "email".to_string(),
        )
        .with_cost(1.0);

        let posts = PlanNode::new("posts".to_string(), "List".to_string(), "posts".to_string())
            .with_cost(10.0)
            .with_child("title".to_string());

        let title = PlanNode::new(
            "title".to_string(),
            "Field".to_string(),
            "title".to_string(),
        )
        .with_cost(2.0);

        plan.add_node(root);
        plan.add_node(user);
        plan.add_node(name);
        plan.add_node(email);
        plan.add_node(posts);
        plan.add_node(title);

        assert_eq!(plan.nodes.len(), 6);
        assert!(plan.total_cost > 0.0);

        let levels = plan.execution_levels();
        assert_eq!(levels.len(), 6);
    }

    #[test]
    fn test_visualizer_without_costs() {
        let plan = create_sample_plan();
        let visualizer = QueryPlanVisualizer::new().with_costs(false);

        let tree = visualizer
            .visualize(&plan, VisualizationFormat::AsciiTree)
            .unwrap();
        assert!(!tree.contains("cost:"));
    }

    #[test]
    fn test_visualizer_without_execution_times() {
        let plan = create_sample_plan();
        let visualizer = QueryPlanVisualizer::new().with_execution_times(false);

        let tree = visualizer
            .visualize(&plan, VisualizationFormat::AsciiTree)
            .unwrap();
        assert!(!tree.contains("time:"));
    }
}
