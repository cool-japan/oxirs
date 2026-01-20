//! Debugging Utilities for SPARQL Queries
//!
//! This module provides comprehensive debugging tools for SPARQL query development,
//! troubleshooting, and performance analysis. It includes query inspection, execution
//! tracing, breakpoints, variable tracking, and plan visualization.
//!
//! # Features
//!
//! - **Query Inspector**: Analyze query structure and patterns
//! - **Execution Debugger**: Step-by-step execution with breakpoints
//! - **Performance Profiler**: Identify bottlenecks and slow operations
//! - **Variable Tracker**: Monitor variable bindings throughout execution
//! - **Plan Visualizer**: Generate visual representations of query plans
//! - **Rewrite Tracker**: Track query rewriting transformations
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_arq::debug_utilities::{QueryDebugger, DebugConfig};
//!
//! let config = DebugConfig::default()
//!     .with_breakpoints(true)
//!     .with_variable_tracking(true);
//!
//! let mut debugger = QueryDebugger::new(config)?;
//!
//! // Set breakpoint
//! debugger.add_breakpoint(DebugBreakpoint::AfterJoin { join_id: 0 });
//!
//! // Execute with debugging
//! let result = debugger.execute_with_debug(&query, &dataset)?;
//!
//! // Inspect execution trace
//! let trace = debugger.get_execution_trace();
//! ```

use crate::algebra::Algebra;
use anyhow::Result;
use scirs2_core::metrics::MetricsRegistry;
use scirs2_core::profiling::Profiler;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;
use tracing::info;

/// Configuration for query debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugConfig {
    /// Enable execution tracing
    pub enable_tracing: bool,

    /// Enable breakpoint support
    pub enable_breakpoints: bool,

    /// Enable variable tracking
    pub enable_variable_tracking: bool,

    /// Enable performance profiling
    pub enable_profiling: bool,

    /// Enable plan visualization
    pub enable_plan_visualization: bool,

    /// Maximum trace entries to keep
    pub max_trace_entries: usize,

    /// Trace detail level (0-3)
    pub trace_detail_level: usize,

    /// Enable query rewrite tracking
    pub track_rewrites: bool,

    /// Enable memory tracking
    pub track_memory: bool,

    /// Output format for visualizations
    pub visualization_format: VisualizationFormat,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            enable_tracing: true,
            enable_breakpoints: true,
            enable_variable_tracking: true,
            enable_profiling: true,
            enable_plan_visualization: true,
            max_trace_entries: 10000,
            trace_detail_level: 2,
            track_rewrites: true,
            track_memory: true,
            visualization_format: VisualizationFormat::Text,
        }
    }
}

impl DebugConfig {
    /// Enable breakpoints
    pub fn with_breakpoints(mut self, enabled: bool) -> Self {
        self.enable_breakpoints = enabled;
        self
    }

    /// Enable variable tracking
    pub fn with_variable_tracking(mut self, enabled: bool) -> Self {
        self.enable_variable_tracking = enabled;
        self
    }

    /// Set trace detail level
    pub fn with_trace_level(mut self, level: usize) -> Self {
        self.trace_detail_level = level.min(3);
        self
    }

    /// Set visualization format
    pub fn with_visualization_format(mut self, format: VisualizationFormat) -> Self {
        self.visualization_format = format;
        self
    }
}

/// Visualization output formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualizationFormat {
    /// Plain text tree structure
    Text,
    /// DOT graph format (Graphviz)
    Dot,
    /// JSON structure
    Json,
    /// Mermaid diagram
    Mermaid,
}

/// Query debugger for SPARQL queries
pub struct QueryDebugger {
    /// Debug configuration
    config: DebugConfig,

    /// Active breakpoints
    breakpoints: Vec<DebugBreakpoint>,

    /// Execution trace
    trace: VecDeque<TraceEntry>,

    /// Variable binding history
    variable_history: HashMap<String, Vec<VariableBinding>>,

    /// Performance profiler (reserved for future use)
    #[allow(dead_code)]
    profiler: Arc<Profiler>,

    /// Metrics registry (reserved for future use)
    #[allow(dead_code)]
    metrics: Arc<MetricsRegistry>,

    /// Query rewrite history
    rewrite_history: Vec<RewriteStep>,

    /// Current execution state
    state: ExecutionState,

    /// Breakpoint hit counter
    breakpoint_hits: usize,
}

/// Debug breakpoint types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DebugBreakpoint {
    /// Break before executing a triple pattern
    BeforeTriplePattern { pattern_id: usize },

    /// Break after executing a triple pattern
    AfterTriplePattern { pattern_id: usize },

    /// Break before a join operation
    BeforeJoin { join_id: usize },

    /// Break after a join operation
    AfterJoin { join_id: usize },

    /// Break when a filter is evaluated
    OnFilter { filter_id: usize },

    /// Break when a variable is bound
    OnVariableBound { variable: String },

    /// Break when result count exceeds threshold
    OnResultCountExceeds { threshold: usize },

    /// Break on execution time threshold
    OnTimeExceeds { threshold: Duration },

    /// Break on memory usage threshold
    OnMemoryExceeds { threshold_bytes: usize },

    /// Conditional breakpoint with custom expression
    Conditional { condition: String },
}

/// Execution trace entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEntry {
    /// Sequence number
    pub seq: usize,

    /// Timestamp
    pub timestamp: Duration,

    /// Operation type
    pub operation: Operation,

    /// Execution time for this operation
    pub duration: Duration,

    /// Number of results at this point
    pub result_count: usize,

    /// Memory usage (bytes)
    pub memory_usage: usize,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Operation types in execution trace
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operation {
    /// Query parsing
    Parse,

    /// Algebra generation
    GenerateAlgebra,

    /// Query optimization
    Optimize,

    /// Plan generation
    GeneratePlan,

    /// Triple pattern scan
    ScanPattern { pattern_id: usize },

    /// Join operation
    Join { join_id: usize, join_type: JoinType },

    /// Filter evaluation
    EvaluateFilter { filter_id: usize },

    /// Projection
    Project { variables: Vec<String> },

    /// Aggregation
    Aggregate { functions: Vec<String> },

    /// Sorting
    Sort { variables: Vec<String> },

    /// Distinct operation
    Distinct,

    /// Limit/Offset
    Slice { offset: usize, limit: usize },

    /// Custom operation
    Custom { name: String },
}

/// Join operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JoinType {
    /// Inner join
    Inner,
    /// Left outer join (OPTIONAL)
    LeftOuter,
    /// Minus
    Minus,
    /// Union
    Union,
}

/// Variable binding at a specific point in execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableBinding {
    /// Variable name
    pub variable: String,

    /// Bound value (as string representation)
    pub value: String,

    /// Binding timestamp
    pub timestamp: Duration,

    /// Source operation
    pub source: String,
}

/// Query rewrite transformation step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewriteStep {
    /// Step number
    pub step: usize,

    /// Rewrite rule applied
    pub rule: String,

    /// Description of transformation
    pub description: String,

    /// Algebra before rewrite
    pub before: String,

    /// Algebra after rewrite
    pub after: String,

    /// Estimated improvement
    pub improvement: Option<f64>,
}

/// Current execution state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionState {
    /// Not started
    Idle,

    /// Currently executing
    Running,

    /// Paused at breakpoint
    Paused,

    /// Execution completed
    Completed,

    /// Execution failed
    Failed,
}

impl QueryDebugger {
    /// Create a new query debugger
    pub fn new(config: DebugConfig) -> Result<Self> {
        let profiler = Arc::new(Profiler::new());
        let metrics = Arc::new(MetricsRegistry::new());

        Ok(Self {
            config,
            breakpoints: Vec::new(),
            trace: VecDeque::new(),
            variable_history: HashMap::new(),
            profiler,
            metrics,
            rewrite_history: Vec::new(),
            state: ExecutionState::Idle,
            breakpoint_hits: 0,
        })
    }

    /// Add a breakpoint
    pub fn add_breakpoint(&mut self, breakpoint: DebugBreakpoint) {
        if self.config.enable_breakpoints {
            info!("Adding breakpoint: {:?}", breakpoint);
            self.breakpoints.push(breakpoint);
        }
    }

    /// Remove a breakpoint
    pub fn remove_breakpoint(&mut self, index: usize) -> Option<DebugBreakpoint> {
        if index < self.breakpoints.len() {
            Some(self.breakpoints.remove(index))
        } else {
            None
        }
    }

    /// Clear all breakpoints
    pub fn clear_breakpoints(&mut self) {
        self.breakpoints.clear();
        info!("Cleared all breakpoints");
    }

    /// Record a trace entry
    pub fn record_trace(&mut self, operation: Operation, duration: Duration, result_count: usize) {
        if !self.config.enable_tracing {
            return;
        }

        let entry = TraceEntry {
            seq: self.trace.len(),
            timestamp: self.get_elapsed_time(),
            operation,
            duration,
            result_count,
            memory_usage: self.estimate_memory_usage(),
            metadata: HashMap::new(),
        };

        self.trace.push_back(entry);

        // Trim trace if it exceeds max entries
        while self.trace.len() > self.config.max_trace_entries {
            self.trace.pop_front();
        }
    }

    /// Track a variable binding
    pub fn track_variable(&mut self, variable: String, value: String, source: String) {
        if !self.config.enable_variable_tracking {
            return;
        }

        let binding = VariableBinding {
            variable: variable.clone(),
            value,
            timestamp: self.get_elapsed_time(),
            source,
        };

        self.variable_history
            .entry(variable.clone())
            .or_default()
            .push(binding);

        // Check for variable binding breakpoint
        if let Some(_bp) = self.breakpoints.iter().find(
            |bp| matches!(bp, DebugBreakpoint::OnVariableBound { variable: v } if v == &variable),
        ) {
            self.hit_breakpoint();
        }
    }

    /// Record a query rewrite step
    pub fn record_rewrite(
        &mut self,
        rule: String,
        description: String,
        before: &Algebra,
        after: &Algebra,
    ) {
        if !self.config.track_rewrites {
            return;
        }

        let step = RewriteStep {
            step: self.rewrite_history.len(),
            rule,
            description,
            before: format!("{:?}", before),
            after: format!("{:?}", after),
            improvement: None, // Could calculate cost difference
        };

        self.rewrite_history.push(step);
    }

    /// Check if a breakpoint should be triggered
    pub fn should_break(&self, operation: &Operation, result_count: usize) -> bool {
        if !self.config.enable_breakpoints {
            return false;
        }

        for bp in &self.breakpoints {
            let should_break = match (bp, operation) {
                (
                    DebugBreakpoint::BeforeTriplePattern { pattern_id: bp_id },
                    Operation::ScanPattern { pattern_id: op_id },
                ) => bp_id == op_id,
                (
                    DebugBreakpoint::BeforeJoin { join_id: bp_id },
                    Operation::Join { join_id: op_id, .. },
                ) => bp_id == op_id,
                (DebugBreakpoint::OnResultCountExceeds { threshold }, _) => {
                    result_count > *threshold
                }
                _ => false,
            };

            if should_break {
                return true;
            }
        }

        false
    }

    /// Handle breakpoint hit
    fn hit_breakpoint(&mut self) {
        self.breakpoint_hits += 1;
        self.state = ExecutionState::Paused;
        info!("Breakpoint hit #{}", self.breakpoint_hits);
    }

    /// Resume execution after breakpoint
    pub fn resume(&mut self) {
        if self.state == ExecutionState::Paused {
            self.state = ExecutionState::Running;
            info!("Resuming execution");
        }
    }

    /// Get execution trace
    pub fn get_execution_trace(&self) -> &VecDeque<TraceEntry> {
        &self.trace
    }

    /// Get variable binding history for a specific variable
    pub fn get_variable_history(&self, variable: &str) -> Option<&Vec<VariableBinding>> {
        self.variable_history.get(variable)
    }

    /// Get all variable histories
    pub fn get_all_variable_histories(&self) -> &HashMap<String, Vec<VariableBinding>> {
        &self.variable_history
    }

    /// Get query rewrite history
    pub fn get_rewrite_history(&self) -> &[RewriteStep] {
        &self.rewrite_history
    }

    /// Get current execution state
    pub fn get_state(&self) -> ExecutionState {
        self.state
    }

    /// Get breakpoint hit count
    pub fn get_breakpoint_hits(&self) -> usize {
        self.breakpoint_hits
    }

    /// Visualize query plan
    pub fn visualize_plan(&self, algebra: &Algebra) -> Result<String> {
        if !self.config.enable_plan_visualization {
            return Ok("Plan visualization disabled".to_string());
        }

        match self.config.visualization_format {
            VisualizationFormat::Text => self.visualize_as_text(algebra),
            VisualizationFormat::Dot => self.visualize_as_dot(algebra),
            VisualizationFormat::Json => self.visualize_as_json(algebra),
            VisualizationFormat::Mermaid => self.visualize_as_mermaid(algebra),
        }
    }

    /// Visualize as text tree
    fn visualize_as_text(&self, algebra: &Algebra) -> Result<String> {
        let mut output = String::new();
        self.visualize_algebra_recursive(algebra, 0, &mut output);
        Ok(output)
    }

    /// Recursive helper for text visualization
    #[allow(clippy::only_used_in_recursion)]
    fn visualize_algebra_recursive(&self, algebra: &Algebra, depth: usize, output: &mut String) {
        let indent = "  ".repeat(depth);

        match algebra {
            Algebra::Bgp(patterns) => {
                output.push_str(&format!("{}BGP ({} patterns)\n", indent, patterns.len()));
            }
            Algebra::Join { left, right } => {
                output.push_str(&format!("{}JOIN\n", indent));
                self.visualize_algebra_recursive(left, depth + 1, output);
                self.visualize_algebra_recursive(right, depth + 1, output);
            }
            Algebra::LeftJoin { left, right, .. } => {
                output.push_str(&format!("{}LEFT JOIN (OPTIONAL)\n", indent));
                self.visualize_algebra_recursive(left, depth + 1, output);
                self.visualize_algebra_recursive(right, depth + 1, output);
            }
            Algebra::Union { left, right } => {
                output.push_str(&format!("{}UNION\n", indent));
                self.visualize_algebra_recursive(left, depth + 1, output);
                self.visualize_algebra_recursive(right, depth + 1, output);
            }
            Algebra::Filter { pattern, condition } => {
                output.push_str(&format!("{}FILTER: {:?}\n", indent, condition));
                self.visualize_algebra_recursive(pattern, depth + 1, output);
            }
            Algebra::Project { pattern, variables } => {
                output.push_str(&format!(
                    "{}PROJECT: [{}]\n",
                    indent,
                    variables
                        .iter()
                        .map(|v| v.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
                self.visualize_algebra_recursive(pattern, depth + 1, output);
            }
            Algebra::Distinct { pattern } => {
                output.push_str(&format!("{}DISTINCT\n", indent));
                self.visualize_algebra_recursive(pattern, depth + 1, output);
            }
            Algebra::OrderBy {
                pattern,
                conditions,
            } => {
                output.push_str(&format!("{}ORDER BY ({} keys)\n", indent, conditions.len()));
                self.visualize_algebra_recursive(pattern, depth + 1, output);
            }
            Algebra::Slice {
                pattern,
                offset,
                limit,
            } => {
                output.push_str(&format!(
                    "{}SLICE (offset={:?}, limit={:?})\n",
                    indent, offset, limit
                ));
                self.visualize_algebra_recursive(pattern, depth + 1, output);
            }
            Algebra::Group { pattern, .. } => {
                output.push_str(&format!("{}GROUP\n", indent));
                self.visualize_algebra_recursive(pattern, depth + 1, output);
            }
            _ => {
                output.push_str(&format!("{}{:?}\n", indent, algebra));
            }
        }
    }

    /// Visualize as DOT graph (Graphviz format)
    fn visualize_as_dot(&self, algebra: &Algebra) -> Result<String> {
        let mut output = String::from("digraph query_plan {\n");
        output.push_str("  rankdir=TB;\n");
        output.push_str("  node [shape=box];\n\n");

        let mut node_id = 0;
        self.visualize_dot_recursive(algebra, &mut node_id, None, &mut output);

        output.push_str("}\n");
        Ok(output)
    }

    /// Recursive helper for DOT visualization
    #[allow(clippy::only_used_in_recursion)]
    fn visualize_dot_recursive(
        &self,
        algebra: &Algebra,
        node_id: &mut usize,
        parent_id: Option<usize>,
        output: &mut String,
    ) -> usize {
        let current_id = *node_id;
        *node_id += 1;

        let label = match algebra {
            Algebra::Bgp(patterns) => format!("BGP\\n{} patterns", patterns.len()),
            Algebra::Join { .. } => "JOIN".to_string(),
            Algebra::LeftJoin { .. } => "LEFT JOIN".to_string(),
            Algebra::Union { .. } => "UNION".to_string(),
            Algebra::Filter { .. } => "FILTER".to_string(),
            Algebra::Project { variables, .. } => {
                format!("PROJECT\\n{} vars", variables.len())
            }
            Algebra::Distinct { .. } => "DISTINCT".to_string(),
            Algebra::OrderBy { .. } => "ORDER BY".to_string(),
            _ => format!("{:?}", algebra)
                .split('{')
                .next()
                .unwrap_or("Unknown")
                .to_string(),
        };

        output.push_str(&format!("  n{} [label=\"{}\"];\n", current_id, label));

        if let Some(pid) = parent_id {
            output.push_str(&format!("  n{} -> n{};\n", pid, current_id));
        }

        // Recurse for child nodes
        match algebra {
            Algebra::Join { left, right }
            | Algebra::LeftJoin { left, right, .. }
            | Algebra::Union { left, right } => {
                self.visualize_dot_recursive(left, node_id, Some(current_id), output);
                self.visualize_dot_recursive(right, node_id, Some(current_id), output);
            }
            Algebra::Filter { pattern, .. }
            | Algebra::Project { pattern, .. }
            | Algebra::Distinct { pattern }
            | Algebra::OrderBy { pattern, .. }
            | Algebra::Slice { pattern, .. }
            | Algebra::Group { pattern, .. } => {
                self.visualize_dot_recursive(pattern, node_id, Some(current_id), output);
            }
            _ => {}
        }

        current_id
    }

    /// Visualize as JSON
    fn visualize_as_json(&self, algebra: &Algebra) -> Result<String> {
        // Simple JSON representation
        Ok(serde_json::to_string_pretty(&format!("{:?}", algebra))?)
    }

    /// Visualize as Mermaid diagram
    fn visualize_as_mermaid(&self, algebra: &Algebra) -> Result<String> {
        let mut output = String::from("graph TD\n");

        let mut node_id = 0;
        self.visualize_mermaid_recursive(algebra, &mut node_id, None, &mut output);

        Ok(output)
    }

    /// Recursive helper for Mermaid visualization
    #[allow(clippy::only_used_in_recursion)]
    fn visualize_mermaid_recursive(
        &self,
        algebra: &Algebra,
        node_id: &mut usize,
        parent_id: Option<usize>,
        output: &mut String,
    ) -> usize {
        let current_id = *node_id;
        *node_id += 1;

        let label = match algebra {
            Algebra::Bgp(patterns) => format!("BGP<br/>{} patterns", patterns.len()),
            Algebra::Join { .. } => "JOIN".to_string(),
            Algebra::Filter { .. } => "FILTER".to_string(),
            _ => format!("{:?}", algebra)
                .split('{')
                .next()
                .unwrap_or("Unknown")
                .to_string(),
        };

        output.push_str(&format!("  N{}[{}]\n", current_id, label));

        if let Some(pid) = parent_id {
            output.push_str(&format!("  N{} --> N{}\n", pid, current_id));
        }

        // Recurse for child nodes
        match algebra {
            Algebra::Join { left, right } | Algebra::Union { left, right } => {
                self.visualize_mermaid_recursive(left, node_id, Some(current_id), output);
                self.visualize_mermaid_recursive(right, node_id, Some(current_id), output);
            }
            Algebra::Filter { pattern, .. } | Algebra::Project { pattern, .. } => {
                self.visualize_mermaid_recursive(pattern, node_id, Some(current_id), output);
            }
            _ => {}
        }

        current_id
    }

    /// Generate debug report
    pub fn generate_report(&self) -> DebugReport {
        DebugReport {
            total_operations: self.trace.len(),
            total_duration: self.get_elapsed_time(),
            breakpoint_hits: self.breakpoint_hits,
            variables_tracked: self.variable_history.len(),
            rewrites_applied: self.rewrite_history.len(),
            state: self.state,
            slowest_operations: self.get_slowest_operations(5),
            memory_peak: self.trace.iter().map(|e| e.memory_usage).max().unwrap_or(0),
        }
    }

    /// Get slowest operations
    fn get_slowest_operations(&self, count: usize) -> Vec<(Operation, Duration)> {
        let mut ops: Vec<_> = self
            .trace
            .iter()
            .map(|e| (e.operation.clone(), e.duration))
            .collect();

        ops.sort_by(|a, b| b.1.cmp(&a.1));
        ops.truncate(count);
        ops
    }

    /// Get elapsed time since start
    fn get_elapsed_time(&self) -> Duration {
        self.trace
            .back()
            .map(|e| e.timestamp)
            .unwrap_or(Duration::ZERO)
    }

    /// Estimate current memory usage
    fn estimate_memory_usage(&self) -> usize {
        // Simple estimation based on trace size and variable history
        let trace_mem = self.trace.len() * std::mem::size_of::<TraceEntry>();
        let var_mem = self.variable_history.len() * 256; // Rough estimate
        trace_mem + var_mem
    }

    /// Reset debugger state
    pub fn reset(&mut self) {
        self.trace.clear();
        self.variable_history.clear();
        self.rewrite_history.clear();
        self.state = ExecutionState::Idle;
        self.breakpoint_hits = 0;
        info!("Debugger reset");
    }
}

/// Debug report summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugReport {
    /// Total operations executed
    pub total_operations: usize,

    /// Total execution duration
    pub total_duration: Duration,

    /// Number of breakpoint hits
    pub breakpoint_hits: usize,

    /// Number of variables tracked
    pub variables_tracked: usize,

    /// Number of rewrites applied
    pub rewrites_applied: usize,

    /// Current execution state
    pub state: ExecutionState,

    /// Slowest operations
    pub slowest_operations: Vec<(Operation, Duration)>,

    /// Peak memory usage
    pub memory_peak: usize,
}

impl fmt::Display for DebugReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Query Debug Report ===")?;
        writeln!(f, "Total Operations: {}", self.total_operations)?;
        writeln!(f, "Total Duration: {:?}", self.total_duration)?;
        writeln!(f, "Breakpoint Hits: {}", self.breakpoint_hits)?;
        writeln!(f, "Variables Tracked: {}", self.variables_tracked)?;
        writeln!(f, "Rewrites Applied: {}", self.rewrites_applied)?;
        writeln!(f, "Execution State: {:?}", self.state)?;
        writeln!(f, "Peak Memory: {} bytes", self.memory_peak)?;
        writeln!(f, "\nSlowest Operations:")?;
        for (i, (op, duration)) in self.slowest_operations.iter().enumerate() {
            writeln!(f, "  {}. {:?} - {:?}", i + 1, op, duration)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debugger_creation() {
        let config = DebugConfig::default();
        let debugger = QueryDebugger::new(config);
        assert!(debugger.is_ok());
    }

    #[test]
    fn test_breakpoint_management() {
        let mut debugger = QueryDebugger::new(DebugConfig::default()).unwrap();

        // Add breakpoint
        debugger.add_breakpoint(DebugBreakpoint::BeforeJoin { join_id: 0 });
        assert_eq!(debugger.breakpoints.len(), 1);

        // Remove breakpoint
        let removed = debugger.remove_breakpoint(0);
        assert!(removed.is_some());
        assert_eq!(debugger.breakpoints.len(), 0);
    }

    #[test]
    fn test_trace_recording() {
        let mut debugger = QueryDebugger::new(DebugConfig::default()).unwrap();

        let op = Operation::Parse;
        debugger.record_trace(op.clone(), Duration::from_millis(10), 0);

        assert_eq!(debugger.trace.len(), 1);
        assert_eq!(debugger.trace[0].operation, op);
    }

    #[test]
    fn test_variable_tracking() {
        let mut debugger = QueryDebugger::new(DebugConfig::default()).unwrap();

        debugger.track_variable(
            "x".to_string(),
            "value1".to_string(),
            "pattern_0".to_string(),
        );

        let history = debugger.get_variable_history("x");
        assert!(history.is_some());
        assert_eq!(history.unwrap().len(), 1);
    }

    #[test]
    fn test_rewrite_tracking() {
        let mut debugger = QueryDebugger::new(DebugConfig::default()).unwrap();

        use crate::algebra::Algebra;

        let before = Algebra::Bgp(vec![]);
        let after = Algebra::Bgp(vec![]);

        debugger.record_rewrite(
            "test_rule".to_string(),
            "Test rewrite".to_string(),
            &before,
            &after,
        );

        assert_eq!(debugger.rewrite_history.len(), 1);
    }

    #[test]
    fn test_execution_state_transitions() {
        let mut debugger = QueryDebugger::new(DebugConfig::default()).unwrap();

        assert_eq!(debugger.get_state(), ExecutionState::Idle);

        debugger.state = ExecutionState::Running;
        assert_eq!(debugger.get_state(), ExecutionState::Running);

        debugger.hit_breakpoint();
        assert_eq!(debugger.get_state(), ExecutionState::Paused);

        debugger.resume();
        assert_eq!(debugger.get_state(), ExecutionState::Running);
    }

    #[test]
    fn test_debug_report_generation() {
        let mut debugger = QueryDebugger::new(DebugConfig::default()).unwrap();

        debugger.record_trace(Operation::Parse, Duration::from_millis(5), 0);
        debugger.record_trace(Operation::GenerateAlgebra, Duration::from_millis(10), 0);

        let report = debugger.generate_report();
        assert_eq!(report.total_operations, 2);
    }

    #[test]
    fn test_config_builder() {
        let config = DebugConfig::default()
            .with_breakpoints(false)
            .with_variable_tracking(true)
            .with_trace_level(3)
            .with_visualization_format(VisualizationFormat::Dot);

        assert!(!config.enable_breakpoints);
        assert!(config.enable_variable_tracking);
        assert_eq!(config.trace_detail_level, 3);
        assert_eq!(config.visualization_format, VisualizationFormat::Dot);
    }

    #[test]
    fn test_debugger_reset() {
        let mut debugger = QueryDebugger::new(DebugConfig::default()).unwrap();

        debugger.record_trace(Operation::Parse, Duration::from_millis(5), 0);
        debugger.track_variable("x".to_string(), "val".to_string(), "src".to_string());

        debugger.reset();

        assert_eq!(debugger.trace.len(), 0);
        assert_eq!(debugger.variable_history.len(), 0);
        assert_eq!(debugger.get_state(), ExecutionState::Idle);
    }
}
