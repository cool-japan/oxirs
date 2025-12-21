//! SPARQL Query Plan Visualization and Debugging Tools
//!
//! This module provides comprehensive tools for visualizing and debugging SPARQL query execution plans.
//! It helps developers understand query optimization decisions and identify performance bottlenecks.
//!
//! # Features
//! - ASCII art tree visualization of query plans
//! - Execution statistics overlay
//! - Cost model visualization
//! - Index usage analysis
//! - Cardinality estimation display
//! - JSON export for external tools
//!
//! # Example
//! ```rust
//! use oxirs_core::query::query_plan_visualizer::QueryPlanVisualizer;
//!
//! let visualizer = QueryPlanVisualizer::new();
//! let plan = create_query_plan();
//! let ascii_tree = visualizer.visualize_as_tree(&plan);
//! println!("{}", ascii_tree);
//! ```

use crate::error::OxirsError;
use crate::Result;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Query plan node representing a step in query execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlanNode {
    /// Node type (e.g., "TriplePattern", "Join", "Filter")
    pub node_type: String,
    /// Description of the operation
    pub description: String,
    /// Estimated cardinality (number of results)
    pub estimated_cardinality: Option<usize>,
    /// Actual cardinality (if executed)
    pub actual_cardinality: Option<usize>,
    /// Estimated cost (arbitrary units)
    pub estimated_cost: Option<f64>,
    /// Actual execution time in microseconds
    pub execution_time_us: Option<u64>,
    /// Index used (if applicable)
    pub index_used: Option<String>,
    /// Child nodes
    pub children: Vec<QueryPlanNode>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl QueryPlanNode {
    /// Create a new query plan node
    pub fn new(node_type: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            node_type: node_type.into(),
            description: description.into(),
            estimated_cardinality: None,
            actual_cardinality: None,
            estimated_cost: None,
            execution_time_us: None,
            index_used: None,
            children: Vec::new(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add a child node
    pub fn add_child(&mut self, child: QueryPlanNode) {
        self.children.push(child);
    }

    /// Set estimated cardinality
    pub fn with_estimated_cardinality(mut self, cardinality: usize) -> Self {
        self.estimated_cardinality = Some(cardinality);
        self
    }

    /// Set actual cardinality
    pub fn with_actual_cardinality(mut self, cardinality: usize) -> Self {
        self.actual_cardinality = Some(cardinality);
        self
    }

    /// Set estimated cost
    pub fn with_estimated_cost(mut self, cost: f64) -> Self {
        self.estimated_cost = Some(cost);
        self
    }

    /// Set execution time
    pub fn with_execution_time(mut self, time_us: u64) -> Self {
        self.execution_time_us = Some(time_us);
        self
    }

    /// Set index used
    pub fn with_index(mut self, index: impl Into<String>) -> Self {
        self.index_used = Some(index.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Query plan visualizer with multiple output formats
pub struct QueryPlanVisualizer {
    /// Show execution statistics if available
    show_stats: bool,
    /// Show cost estimates
    show_costs: bool,
    /// Show index usage
    show_indexes: bool,
    /// Show cardinality estimates vs actuals
    show_cardinality: bool,
}

impl Default for QueryPlanVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryPlanVisualizer {
    /// Create a new visualizer with default settings
    pub fn new() -> Self {
        Self {
            show_stats: true,
            show_costs: true,
            show_indexes: true,
            show_cardinality: true,
        }
    }

    /// Enable/disable statistics display
    pub fn with_stats(mut self, show: bool) -> Self {
        self.show_stats = show;
        self
    }

    /// Enable/disable cost display
    pub fn with_costs(mut self, show: bool) -> Self {
        self.show_costs = show;
        self
    }

    /// Enable/disable index display
    pub fn with_indexes(mut self, show: bool) -> Self {
        self.show_indexes = show;
        self
    }

    /// Enable/disable cardinality display
    pub fn with_cardinality(mut self, show: bool) -> Self {
        self.show_cardinality = show;
        self
    }

    /// Visualize query plan as ASCII tree
    pub fn visualize_as_tree(&self, plan: &QueryPlanNode) -> String {
        let mut output = String::new();
        self.render_node(plan, &mut output, "", true);
        output
    }

    /// Render a single node with tree structure
    fn render_node(&self, node: &QueryPlanNode, output: &mut String, prefix: &str, is_last: bool) {
        // Draw tree structure
        output.push_str(prefix);
        if is_last {
            output.push_str("└─ ");
        } else {
            output.push_str("├─ ");
        }

        // Node type and description
        output.push_str(&format!("[{}] {}", node.node_type, node.description));

        // Add statistics inline
        let mut stats = Vec::new();

        if self.show_cardinality {
            if let Some(est) = node.estimated_cardinality {
                if let Some(act) = node.actual_cardinality {
                    stats.push(format!("card: {} → {}", est, act));
                } else {
                    stats.push(format!("card: ~{}", est));
                }
            }
        }

        if self.show_costs {
            if let Some(cost) = node.estimated_cost {
                stats.push(format!("cost: {:.2}", cost));
            }
        }

        if self.show_stats {
            if let Some(time) = node.execution_time_us {
                if time >= 1000 {
                    stats.push(format!("time: {:.2}ms", time as f64 / 1000.0));
                } else {
                    stats.push(format!("time: {}μs", time));
                }
            }
        }

        if self.show_indexes {
            if let Some(idx) = &node.index_used {
                stats.push(format!("index: {}", idx));
            }
        }

        if !stats.is_empty() {
            output.push_str(&format!(" ({})", stats.join(", ")));
        }

        output.push('\n');

        // Render children
        let child_prefix = if is_last {
            format!("{}   ", prefix)
        } else {
            format!("{}│  ", prefix)
        };

        for (i, child) in node.children.iter().enumerate() {
            let is_last_child = i == node.children.len() - 1;
            self.render_node(child, output, &child_prefix, is_last_child);
        }
    }

    /// Export query plan as JSON
    pub fn export_as_json(&self, plan: &QueryPlanNode) -> Result<String> {
        serde_json::to_string_pretty(plan)
            .map_err(|e| OxirsError::Query(format!("Failed to serialize query plan: {}", e)))
    }

    /// Generate execution summary
    pub fn generate_summary(&self, plan: &QueryPlanNode) -> QueryPlanSummary {
        let mut summary = QueryPlanSummary::default();
        Self::collect_stats(plan, &mut summary);
        summary
    }

    /// Recursively collect statistics
    fn collect_stats(node: &QueryPlanNode, summary: &mut QueryPlanSummary) {
        summary.total_nodes += 1;

        if let Some(time) = node.execution_time_us {
            summary.total_execution_time_us += time;
        }

        if let Some(cost) = node.estimated_cost {
            summary.total_estimated_cost += cost;
        }

        if let Some(est) = node.estimated_cardinality {
            if let Some(act) = node.actual_cardinality {
                let error = (est as f64 - act as f64).abs() / act.max(1) as f64;
                summary.cardinality_errors.push(error);
            }
        }

        if node.index_used.is_some() {
            summary.index_operations += 1;
        }

        match node.node_type.as_str() {
            "TriplePattern" => summary.triple_patterns += 1,
            "Join" => summary.joins += 1,
            "Filter" => summary.filters += 1,
            "Union" => summary.unions += 1,
            "Optional" => summary.optionals += 1,
            _ => {}
        }

        for child in &node.children {
            Self::collect_stats(child, summary);
        }
    }

    /// Identify potential optimizations
    pub fn suggest_optimizations(&self, plan: &QueryPlanNode) -> Vec<OptimizationHint> {
        let mut hints = Vec::new();
        Self::analyze_node(plan, &mut hints);
        hints
    }

    /// Analyze a node for optimization opportunities
    fn analyze_node(node: &QueryPlanNode, hints: &mut Vec<OptimizationHint>) {
        // Check for cardinality misestimation
        if let (Some(est), Some(act)) = (node.estimated_cardinality, node.actual_cardinality) {
            if est > 0 && act > 0 {
                let ratio = est as f64 / act as f64;
                if !(0.1..=10.0).contains(&ratio) {
                    hints.push(OptimizationHint {
                        severity: HintSeverity::Warning,
                        node_type: node.node_type.clone(),
                        description: node.description.clone(),
                        message: format!(
                            "Cardinality misestimation: estimated {} but got {} ({}x off)",
                            est,
                            act,
                            if ratio > 1.0 { ratio } else { 1.0 / ratio }
                        ),
                        suggestion: "Consider updating statistics or adding histogram data"
                            .to_string(),
                    });
                }
            }
        }

        // Check for missing indexes
        if node.node_type == "TriplePattern" && node.index_used.is_none() {
            if let Some(card) = node.actual_cardinality {
                if card > 1000 {
                    hints.push(OptimizationHint {
                        severity: HintSeverity::Info,
                        node_type: node.node_type.clone(),
                        description: node.description.clone(),
                        message: format!("Full scan on {} results without index", card),
                        suggestion: "Consider adding an index for this pattern".to_string(),
                    });
                }
            }
        }

        // Check for expensive operations
        if let Some(time) = node.execution_time_us {
            if time > 100_000 {
                // > 100ms
                hints.push(OptimizationHint {
                    severity: HintSeverity::Warning,
                    node_type: node.node_type.clone(),
                    description: node.description.clone(),
                    message: format!("Slow operation: {:.2}ms", time as f64 / 1000.0),
                    suggestion: "Consider optimizing this operation or adding caching".to_string(),
                });
            }
        }

        // Recursively analyze children
        for child in &node.children {
            Self::analyze_node(child, hints);
        }
    }
}

/// Summary statistics for query plan
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct QueryPlanSummary {
    /// Total number of nodes in plan
    pub total_nodes: usize,
    /// Total execution time in microseconds
    pub total_execution_time_us: u64,
    /// Total estimated cost
    pub total_estimated_cost: f64,
    /// Number of triple patterns
    pub triple_patterns: usize,
    /// Number of joins
    pub joins: usize,
    /// Number of filters
    pub filters: usize,
    /// Number of unions
    pub unions: usize,
    /// Number of optional patterns
    pub optionals: usize,
    /// Number of operations using indexes
    pub index_operations: usize,
    /// Cardinality estimation errors (ratio of estimate/actual)
    pub cardinality_errors: Vec<f64>,
}

impl QueryPlanSummary {
    /// Calculate average cardinality error
    pub fn avg_cardinality_error(&self) -> f64 {
        if self.cardinality_errors.is_empty() {
            0.0
        } else {
            self.cardinality_errors.iter().sum::<f64>() / self.cardinality_errors.len() as f64
        }
    }

    /// Get execution time in milliseconds
    pub fn execution_time_ms(&self) -> f64 {
        self.total_execution_time_us as f64 / 1000.0
    }
}

impl fmt::Display for QueryPlanSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Query Plan Summary:")?;
        writeln!(f, "  Nodes:           {}", self.total_nodes)?;
        writeln!(f, "  Triple Patterns: {}", self.triple_patterns)?;
        writeln!(f, "  Joins:           {}", self.joins)?;
        writeln!(f, "  Filters:         {}", self.filters)?;
        writeln!(f, "  Unions:          {}", self.unions)?;
        writeln!(f, "  Optionals:       {}", self.optionals)?;
        writeln!(f, "  Index Ops:       {}", self.index_operations)?;
        writeln!(f, "  Estimated Cost:  {:.2}", self.total_estimated_cost)?;
        writeln!(f, "  Execution Time:  {:.2}ms", self.execution_time_ms())?;
        if !self.cardinality_errors.is_empty() {
            writeln!(
                f,
                "  Avg Card Error:  {:.2}%",
                self.avg_cardinality_error() * 100.0
            )?;
        }
        Ok(())
    }
}

/// Optimization hint for improving query performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationHint {
    /// Severity level
    pub severity: HintSeverity,
    /// Node type
    pub node_type: String,
    /// Node description
    pub description: String,
    /// Problem description
    pub message: String,
    /// Suggested fix
    pub suggestion: String,
}

impl fmt::Display for OptimizationHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:?}] {} - {}: {} → {}",
            self.severity, self.node_type, self.description, self.message, self.suggestion
        )
    }
}

/// Severity level for optimization hints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HintSeverity {
    /// Informational hint
    Info,
    /// Warning about potential issue
    Warning,
    /// Critical performance problem
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_sample_plan() -> QueryPlanNode {
        let mut root = QueryPlanNode::new("Join", "HashJoin on ?person")
            .with_estimated_cardinality(100)
            .with_actual_cardinality(95)
            .with_estimated_cost(150.0)
            .with_execution_time(5000);

        let child1 = QueryPlanNode::new("TriplePattern", "?person rdf:type foaf:Person")
            .with_estimated_cardinality(1000)
            .with_actual_cardinality(1000)
            .with_index("SPO".to_string())
            .with_execution_time(1000);

        let child2 = QueryPlanNode::new("TriplePattern", "?person foaf:name ?name")
            .with_estimated_cardinality(100)
            .with_actual_cardinality(95)
            .with_index("SPO".to_string())
            .with_execution_time(500);

        root.add_child(child1);
        root.add_child(child2);
        root
    }

    #[test]
    fn test_query_plan_node_creation() {
        let node = QueryPlanNode::new("TriplePattern", "?s ?p ?o")
            .with_estimated_cardinality(1000)
            .with_estimated_cost(100.0);

        assert_eq!(node.node_type, "TriplePattern");
        assert_eq!(node.description, "?s ?p ?o");
        assert_eq!(node.estimated_cardinality, Some(1000));
        assert_eq!(node.estimated_cost, Some(100.0));
    }

    #[test]
    fn test_visualizer_tree_output() {
        let plan = create_sample_plan();
        let visualizer = QueryPlanVisualizer::new();
        let tree = visualizer.visualize_as_tree(&plan);

        assert!(tree.contains("[Join] HashJoin on ?person"));
        assert!(tree.contains("TriplePattern"));
        assert!(tree.contains("foaf:Person"));
        assert!(tree.contains("foaf:name"));
    }

    #[test]
    fn test_json_export() {
        let plan = create_sample_plan();
        let visualizer = QueryPlanVisualizer::new();
        let json = visualizer.export_as_json(&plan).unwrap();

        assert!(json.contains("\"node_type\": \"Join\""));
        assert!(json.contains("\"estimated_cardinality\": 100"));
        assert!(json.contains("\"children\""));
    }

    #[test]
    fn test_summary_generation() {
        let plan = create_sample_plan();
        let visualizer = QueryPlanVisualizer::new();
        let summary = visualizer.generate_summary(&plan);

        assert_eq!(summary.total_nodes, 3); // 1 join + 2 triple patterns
        assert_eq!(summary.joins, 1);
        assert_eq!(summary.triple_patterns, 2);
        assert_eq!(summary.index_operations, 2);
        assert!(summary.total_execution_time_us > 0);
    }

    #[test]
    fn test_optimization_hints() {
        let plan = QueryPlanNode::new("TriplePattern", "?s ?p ?o")
            .with_estimated_cardinality(100)
            .with_actual_cardinality(10000); // 100x underestimate

        let visualizer = QueryPlanVisualizer::new();
        let hints = visualizer.suggest_optimizations(&plan);

        assert!(!hints.is_empty());
        assert!(hints
            .iter()
            .any(|h| h.message.contains("Cardinality misestimation")));
    }

    #[test]
    fn test_slow_operation_detection() {
        let plan = QueryPlanNode::new("Join", "Complex join").with_execution_time(200_000); // 200ms - slow

        let visualizer = QueryPlanVisualizer::new();
        let hints = visualizer.suggest_optimizations(&plan);

        assert!(hints.iter().any(|h| h.message.contains("Slow operation")));
    }

    #[test]
    fn test_visualizer_options() {
        let plan = create_sample_plan();
        let visualizer = QueryPlanVisualizer::new()
            .with_costs(false)
            .with_indexes(false);

        let tree = visualizer.visualize_as_tree(&plan);

        // Should not contain cost or index info
        assert!(!tree.contains("cost:"));
        assert!(!tree.contains("index:"));
        // But should still contain cardinality and time
        assert!(tree.contains("card:"));
        assert!(tree.contains("time:") || tree.contains("μs"));
    }

    #[test]
    fn test_summary_display() {
        let plan = create_sample_plan();
        let visualizer = QueryPlanVisualizer::new();
        let summary = visualizer.generate_summary(&plan);

        let display = format!("{}", summary);
        assert!(display.contains("Query Plan Summary:"));
        assert!(display.contains("Triple Patterns:"));
        assert!(display.contains("Joins:"));
    }
}
