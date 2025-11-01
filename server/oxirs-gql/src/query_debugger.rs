//! Query debugging and analysis tools
//!
//! This module provides comprehensive debugging capabilities:
//! - Query explanation and execution plans
//! - Performance profiling
//! - SPARQL translation preview
//! - Query complexity analysis
//! - Execution tracing

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Query execution phase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionPhase {
    /// Parsing GraphQL query
    Parsing,
    /// Validating query
    Validation,
    /// Translating to SPARQL
    Translation,
    /// Executing SPARQL query
    Execution,
    /// Resolving fields
    Resolution,
    /// Formatting response
    Formatting,
}

impl std::fmt::Display for ExecutionPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionPhase::Parsing => write!(f, "Parsing"),
            ExecutionPhase::Validation => write!(f, "Validation"),
            ExecutionPhase::Translation => write!(f, "Translation"),
            ExecutionPhase::Execution => write!(f, "Execution"),
            ExecutionPhase::Resolution => write!(f, "Resolution"),
            ExecutionPhase::Formatting => write!(f, "Formatting"),
        }
    }
}

/// Timing information for an execution phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTimings {
    pub phase: ExecutionPhase,
    pub duration: Duration,
    pub start_time: u64,
    pub end_time: u64,
}

impl PhaseTimings {
    pub fn new(phase: ExecutionPhase, duration: Duration, start_time: u64) -> Self {
        Self {
            phase,
            duration,
            start_time,
            end_time: start_time + duration.as_millis() as u64,
        }
    }
}

/// Query execution trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    /// Query that was executed
    pub query: String,
    /// Variables used
    pub variables: HashMap<String, serde_json::Value>,
    /// Phase timings
    pub timings: Vec<PhaseTimings>,
    /// Total execution time
    pub total_duration: Duration,
    /// Generated SPARQL queries
    pub sparql_queries: Vec<String>,
    /// Number of results returned
    pub result_count: usize,
    /// Errors encountered
    pub errors: Vec<String>,
    /// Warnings
    pub warnings: Vec<String>,
}

impl ExecutionTrace {
    pub fn new(query: String) -> Self {
        Self {
            query,
            variables: HashMap::new(),
            timings: Vec::new(),
            total_duration: Duration::from_secs(0),
            sparql_queries: Vec::new(),
            result_count: 0,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn add_timing(&mut self, phase: ExecutionPhase, duration: Duration, start_time: u64) {
        self.timings
            .push(PhaseTimings::new(phase, duration, start_time));
    }

    pub fn add_sparql_query(&mut self, query: String) {
        self.sparql_queries.push(query);
    }

    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }

    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Generate a summary report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Query Execution Trace ===\n\n");

        // Query
        report.push_str("Query:\n");
        report.push_str(&format!("{}\n\n", self.query));

        // Variables
        if !self.variables.is_empty() {
            report.push_str("Variables:\n");
            for (key, value) in &self.variables {
                report.push_str(&format!("  {}: {}\n", key, value));
            }
            report.push('\n');
        }

        // Timings
        report.push_str("Execution Timings:\n");
        for timing in &self.timings {
            report.push_str(&format!(
                "  {:<15} {:>8.2}ms\n",
                format!("{}:", timing.phase),
                timing.duration.as_secs_f64() * 1000.0
            ));
        }
        report.push_str(&format!(
            "  {:<15} {:>8.2}ms\n\n",
            "Total:",
            self.total_duration.as_secs_f64() * 1000.0
        ));

        // SPARQL queries
        if !self.sparql_queries.is_empty() {
            report.push_str("Generated SPARQL Queries:\n");
            for (idx, query) in self.sparql_queries.iter().enumerate() {
                report.push_str(&format!("\nQuery #{}:\n", idx + 1));
                report.push_str(&format!("{}\n", query));
            }
            report.push('\n');
        }

        // Results
        report.push_str(&format!("Results: {} records\n\n", self.result_count));

        // Errors
        if !self.errors.is_empty() {
            report.push_str("Errors:\n");
            for error in &self.errors {
                report.push_str(&format!("  - {}\n", error));
            }
            report.push('\n');
        }

        // Warnings
        if !self.warnings.is_empty() {
            report.push_str("Warnings:\n");
            for warning in &self.warnings {
                report.push_str(&format!("  - {}\n", warning));
            }
            report.push('\n');
        }

        report
    }
}

/// Query complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    /// Total complexity score
    pub total_score: u32,
    /// Maximum depth
    pub max_depth: u32,
    /// Number of fields
    pub field_count: u32,
    /// Number of lists
    pub list_count: u32,
    /// Number of fragments
    pub fragment_count: u32,
    /// Field-level complexity scores
    pub field_scores: HashMap<String, u32>,
}

impl ComplexityMetrics {
    pub fn new() -> Self {
        Self {
            total_score: 0,
            max_depth: 0,
            field_count: 0,
            list_count: 0,
            fragment_count: 0,
            field_scores: HashMap::new(),
        }
    }

    pub fn add_field_score(&mut self, field: String, score: u32) {
        self.field_scores.insert(field, score);
        self.total_score += score;
    }

    /// Check if complexity exceeds threshold
    pub fn exceeds_threshold(&self, threshold: u32) -> bool {
        self.total_score > threshold
    }

    /// Generate complexity report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Query Complexity Analysis ===\n\n");
        report.push_str(&format!("Total Complexity Score: {}\n", self.total_score));
        report.push_str(&format!("Maximum Depth: {}\n", self.max_depth));
        report.push_str(&format!("Field Count: {}\n", self.field_count));
        report.push_str(&format!("List Count: {}\n", self.list_count));
        report.push_str(&format!("Fragment Count: {}\n\n", self.fragment_count));

        if !self.field_scores.is_empty() {
            report.push_str("Field-level Complexity:\n");
            let mut sorted_fields: Vec<_> = self.field_scores.iter().collect();
            sorted_fields.sort_by(|a, b| b.1.cmp(a.1));

            for (field, score) in sorted_fields.iter().take(10) {
                report.push_str(&format!("  {:<30} {}\n", field, score));
            }
        }

        report
    }
}

impl Default for ComplexityMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Query explanation with execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryExplanation {
    /// GraphQL query
    pub query: String,
    /// Execution plan steps
    pub plan_steps: Vec<PlanStep>,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Recommendations for optimization
    pub recommendations: Vec<String>,
}

impl QueryExplanation {
    pub fn new(query: String) -> Self {
        Self {
            query,
            plan_steps: Vec::new(),
            estimated_cost: 0.0,
            recommendations: Vec::new(),
        }
    }

    pub fn add_step(&mut self, step: PlanStep) {
        self.estimated_cost += step.estimated_cost;
        self.plan_steps.push(step);
    }

    pub fn add_recommendation(&mut self, recommendation: String) {
        self.recommendations.push(recommendation);
    }

    /// Generate explanation report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Query Execution Plan ===\n\n");
        report.push_str("Query:\n");
        report.push_str(&format!("{}\n\n", self.query));

        report.push_str("Execution Steps:\n");
        for (idx, step) in self.plan_steps.iter().enumerate() {
            report.push_str(&format!(
                "{}. {} (cost: {:.2})\n",
                idx + 1,
                step.description,
                step.estimated_cost
            ));
            if !step.details.is_empty() {
                report.push_str(&format!("   {}\n", step.details));
            }
        }

        report.push_str(&format!(
            "\nTotal Estimated Cost: {:.2}\n\n",
            self.estimated_cost
        ));

        if !self.recommendations.is_empty() {
            report.push_str("Optimization Recommendations:\n");
            for (idx, rec) in self.recommendations.iter().enumerate() {
                report.push_str(&format!("  {}. {}\n", idx + 1, rec));
            }
        }

        report
    }
}

/// Execution plan step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// Step description
    pub description: String,
    /// Step details
    pub details: String,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Expected result count
    pub expected_results: Option<usize>,
}

impl PlanStep {
    pub fn new(description: String, estimated_cost: f64) -> Self {
        Self {
            description,
            details: String::new(),
            estimated_cost,
            expected_results: None,
        }
    }

    pub fn with_details(mut self, details: String) -> Self {
        self.details = details;
        self
    }

    pub fn with_expected_results(mut self, count: usize) -> Self {
        self.expected_results = Some(count);
        self
    }
}

/// Query debugger for profiling and analysis
pub struct QueryDebugger {
    /// Enable tracing
    pub enable_tracing: bool,
    /// Enable profiling
    pub enable_profiling: bool,
    /// Enable SPARQL preview
    pub enable_sparql_preview: bool,
    /// Complexity threshold
    pub complexity_threshold: u32,
}

impl QueryDebugger {
    pub fn new() -> Self {
        Self {
            enable_tracing: true,
            enable_profiling: true,
            enable_sparql_preview: true,
            complexity_threshold: 1000,
        }
    }

    pub fn with_tracing(mut self, enabled: bool) -> Self {
        self.enable_tracing = enabled;
        self
    }

    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.enable_profiling = enabled;
        self
    }

    pub fn with_sparql_preview(mut self, enabled: bool) -> Self {
        self.enable_sparql_preview = enabled;
        self
    }

    pub fn with_complexity_threshold(mut self, threshold: u32) -> Self {
        self.complexity_threshold = threshold;
        self
    }

    /// Start a new execution trace
    pub fn start_trace(&self, query: String) -> ExecutionTrace {
        ExecutionTrace::new(query)
    }

    /// Analyze query complexity
    pub fn analyze_complexity(&self, _query: &str) -> ComplexityMetrics {
        // Simplified implementation - in production, this would parse the query
        // and calculate actual complexity
        ComplexityMetrics::new()
    }

    /// Generate query explanation
    pub fn explain_query(&self, query: String) -> QueryExplanation {
        let mut explanation = QueryExplanation::new(query);

        // Simplified implementation - in production, this would analyze the query
        explanation.add_step(
            PlanStep::new("Parse GraphQL query".to_string(), 1.0)
                .with_details("Parse query into AST".to_string()),
        );

        explanation.add_step(
            PlanStep::new("Validate query".to_string(), 2.0)
                .with_details("Validate against schema".to_string()),
        );

        explanation.add_step(
            PlanStep::new("Translate to SPARQL".to_string(), 3.0)
                .with_details("Generate SPARQL query from GraphQL".to_string()),
        );

        explanation.add_step(
            PlanStep::new("Execute SPARQL query".to_string(), 10.0)
                .with_details("Execute against RDF store".to_string())
                .with_expected_results(100),
        );

        explanation
            .add_recommendation("Consider adding pagination to limit result set".to_string());

        explanation
    }
}

impl Default for QueryDebugger {
    fn default() -> Self {
        Self::new()
    }
}

/// Timer for measuring execution phases
pub struct PhaseTimer {
    start: Instant,
    phase: ExecutionPhase,
}

impl PhaseTimer {
    pub fn start(phase: ExecutionPhase) -> Self {
        Self {
            start: Instant::now(),
            phase,
        }
    }

    pub fn finish(self) -> (ExecutionPhase, Duration) {
        (self.phase, self.start.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_phase_display() {
        assert_eq!(ExecutionPhase::Parsing.to_string(), "Parsing");
        assert_eq!(ExecutionPhase::Validation.to_string(), "Validation");
        assert_eq!(ExecutionPhase::Execution.to_string(), "Execution");
    }

    #[test]
    fn test_execution_trace() {
        let mut trace = ExecutionTrace::new("{ user { name } }".to_string());
        trace.add_timing(ExecutionPhase::Parsing, Duration::from_millis(10), 0);
        trace.add_timing(ExecutionPhase::Execution, Duration::from_millis(50), 10);
        trace.add_sparql_query("SELECT ?name WHERE { ?user foaf:name ?name }".to_string());
        trace.result_count = 42;

        assert_eq!(trace.timings.len(), 2);
        assert_eq!(trace.sparql_queries.len(), 1);
        assert_eq!(trace.result_count, 42);
    }

    #[test]
    fn test_execution_trace_report() {
        let mut trace = ExecutionTrace::new("{ user { name } }".to_string());
        trace.add_timing(ExecutionPhase::Parsing, Duration::from_millis(10), 0);
        trace.total_duration = Duration::from_millis(100);

        let report = trace.generate_report();
        assert!(report.contains("Query Execution Trace"));
        assert!(report.contains("user"));
        assert!(report.contains("Parsing"));
    }

    #[test]
    fn test_complexity_metrics() {
        let mut metrics = ComplexityMetrics::new();
        metrics.add_field_score("user".to_string(), 10);
        metrics.add_field_score("posts".to_string(), 20);
        metrics.max_depth = 3;
        metrics.field_count = 5;

        assert_eq!(metrics.total_score, 30);
        assert_eq!(metrics.field_count, 5);
        assert!(!metrics.exceeds_threshold(100));
        assert!(metrics.exceeds_threshold(20));
    }

    #[test]
    fn test_complexity_report() {
        let mut metrics = ComplexityMetrics::new();
        metrics.add_field_score("user.posts".to_string(), 50);
        metrics.total_score = 50;

        let report = metrics.generate_report();
        assert!(report.contains("Complexity Analysis"));
        assert!(report.contains("50"));
    }

    #[test]
    fn test_query_explanation() {
        let mut explanation = QueryExplanation::new("{ user { name } }".to_string());
        explanation.add_step(PlanStep::new("Parse query".to_string(), 1.0));
        explanation.add_step(PlanStep::new("Execute query".to_string(), 10.0));
        explanation.add_recommendation("Use caching".to_string());

        assert_eq!(explanation.plan_steps.len(), 2);
        assert_eq!(explanation.estimated_cost, 11.0);
        assert_eq!(explanation.recommendations.len(), 1);
    }

    #[test]
    fn test_query_explanation_report() {
        let mut explanation = QueryExplanation::new("{ user { name } }".to_string());
        explanation.add_step(PlanStep::new("Parse".to_string(), 1.0));

        let report = explanation.generate_report();
        assert!(report.contains("Execution Plan"));
        assert!(report.contains("Parse"));
    }

    #[test]
    fn test_plan_step() {
        let step = PlanStep::new("Execute SPARQL".to_string(), 10.0)
            .with_details("Query RDF store".to_string())
            .with_expected_results(100);

        assert_eq!(step.description, "Execute SPARQL");
        assert_eq!(step.estimated_cost, 10.0);
        assert_eq!(step.expected_results, Some(100));
    }

    #[test]
    fn test_query_debugger_creation() {
        let debugger = QueryDebugger::new()
            .with_tracing(true)
            .with_profiling(true)
            .with_complexity_threshold(500);

        assert!(debugger.enable_tracing);
        assert!(debugger.enable_profiling);
        assert_eq!(debugger.complexity_threshold, 500);
    }

    #[test]
    fn test_query_debugger_trace() {
        let debugger = QueryDebugger::new();
        let trace = debugger.start_trace("{ user { name } }".to_string());

        assert_eq!(trace.query, "{ user { name } }");
        assert_eq!(trace.timings.len(), 0);
    }

    #[test]
    fn test_query_debugger_complexity() {
        let debugger = QueryDebugger::new();
        let metrics = debugger.analyze_complexity("{ user { name } }");

        assert_eq!(metrics.total_score, 0); // Simplified implementation
    }

    #[test]
    fn test_query_debugger_explain() {
        let debugger = QueryDebugger::new();
        let explanation = debugger.explain_query("{ user { name } }".to_string());

        assert!(!explanation.plan_steps.is_empty());
        assert!(explanation.estimated_cost > 0.0);
    }

    #[test]
    fn test_phase_timer() {
        let timer = PhaseTimer::start(ExecutionPhase::Parsing);
        std::thread::sleep(Duration::from_millis(10));
        let (phase, duration) = timer.finish();

        assert_eq!(phase, ExecutionPhase::Parsing);
        assert!(duration.as_millis() >= 10);
    }
}
