//! Debug Query Execution Plans
//!
//! Provides detailed debugging capabilities for GraphQL query execution,
//! including execution plan visualization and step-by-step tracing.
//!
//! # Features
//!
//! - Execution plan generation and visualization
//! - Step-by-step query execution tracing
//! - Field resolver timing and performance
//! - Data source query inspection
//! - Variable interpolation tracking
//! - Error propagation analysis
//! - Execution tree visualization

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Execution step type
#[derive(Debug, Clone, PartialEq)]
pub enum StepType {
    Parse,
    Validate,
    Optimize,
    ResolveField {
        field_name: String,
        parent_type: String,
    },
    ExecuteDataSource {
        query: String,
        source: String,
    },
    Transform,
    Serialize,
}

/// Execution step with timing
#[derive(Debug, Clone)]
pub struct ExecutionStep {
    pub step_id: usize,
    pub step_type: StepType,
    pub started_at: Instant,
    pub duration: Option<Duration>,
    pub status: StepStatus,
    pub input: Option<String>,
    pub output: Option<String>,
    pub error: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// Step execution status
#[derive(Debug, Clone, PartialEq)]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
}

/// Query execution plan
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub query_id: String,
    pub query_text: String,
    pub variables: HashMap<String, String>,
    pub steps: Vec<ExecutionStep>,
    pub total_duration: Option<Duration>,
    pub created_at: Instant,
}

impl ExecutionPlan {
    /// Create a new execution plan
    pub fn new(query_id: String, query_text: String, variables: HashMap<String, String>) -> Self {
        Self {
            query_id,
            query_text,
            variables,
            steps: Vec::new(),
            total_duration: None,
            created_at: Instant::now(),
        }
    }

    /// Add a step to the plan
    pub fn add_step(&mut self, step_type: StepType) -> usize {
        let step_id = self.steps.len();
        self.steps.push(ExecutionStep {
            step_id,
            step_type,
            started_at: Instant::now(),
            duration: None,
            status: StepStatus::Pending,
            input: None,
            output: None,
            error: None,
            metadata: HashMap::new(),
        });
        step_id
    }

    /// Start a step
    pub fn start_step(&mut self, step_id: usize) {
        if let Some(step) = self.steps.get_mut(step_id) {
            step.status = StepStatus::Running;
            step.started_at = Instant::now();
        }
    }

    /// Complete a step
    pub fn complete_step(&mut self, step_id: usize, output: Option<String>) {
        if let Some(step) = self.steps.get_mut(step_id) {
            step.duration = Some(step.started_at.elapsed());
            step.status = StepStatus::Completed;
            step.output = output;
        }
    }

    /// Fail a step
    pub fn fail_step(&mut self, step_id: usize, error: String) {
        if let Some(step) = self.steps.get_mut(step_id) {
            step.duration = Some(step.started_at.elapsed());
            step.status = StepStatus::Failed;
            step.error = Some(error);
        }
    }

    /// Skip a step
    pub fn skip_step(&mut self, step_id: usize, reason: String) {
        if let Some(step) = self.steps.get_mut(step_id) {
            step.status = StepStatus::Skipped;
            step.metadata.insert("skip_reason".to_string(), reason);
        }
    }

    /// Set step metadata
    pub fn set_step_metadata(&mut self, step_id: usize, key: String, value: String) {
        if let Some(step) = self.steps.get_mut(step_id) {
            step.metadata.insert(key, value);
        }
    }

    /// Finalize the execution plan
    pub fn finalize(&mut self) {
        self.total_duration = Some(self.created_at.elapsed());
    }

    /// Get execution summary
    pub fn summary(&self) -> ExecutionSummary {
        let mut completed = 0;
        let mut failed = 0;
        let mut skipped = 0;
        let mut total_time = Duration::ZERO;

        for step in &self.steps {
            match step.status {
                StepStatus::Completed => {
                    completed += 1;
                    if let Some(duration) = step.duration {
                        total_time += duration;
                    }
                }
                StepStatus::Failed => failed += 1,
                StepStatus::Skipped => skipped += 1,
                _ => {}
            }
        }

        ExecutionSummary {
            total_steps: self.steps.len(),
            completed_steps: completed,
            failed_steps: failed,
            skipped_steps: skipped,
            total_duration: self.total_duration.unwrap_or(total_time),
        }
    }

    /// Generate ASCII tree visualization
    pub fn visualize_tree(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("Query: {} ({})\n", self.query_id, self.query_text));
        output.push_str("Execution Steps:\n");

        for (i, step) in self.steps.iter().enumerate() {
            let prefix = if i == self.steps.len() - 1 {
                "└─"
            } else {
                "├─"
            };

            let status_icon = match step.status {
                StepStatus::Completed => "✓",
                StepStatus::Failed => "✗",
                StepStatus::Skipped => "○",
                StepStatus::Running => "→",
                StepStatus::Pending => "◦",
            };

            let duration_str = step
                .duration
                .map(|d| format!(" [{:.2}ms]", d.as_secs_f64() * 1000.0))
                .unwrap_or_default();

            output.push_str(&format!(
                "{} {} {:?}{}\n",
                prefix, status_icon, step.step_type, duration_str
            ));

            if let Some(ref error) = step.error {
                output.push_str(&format!("  Error: {}\n", error));
            }
        }

        output
    }

    /// Export as JSON
    pub fn to_json(&self) -> String {
        let mut json = String::from("{");
        json.push_str(&format!("\"query_id\":\"{}\",", self.query_id));
        json.push_str(&format!(
            "\"query_text\":\"{}\",",
            self.query_text.replace('"', "\\\"")
        ));

        // Steps
        json.push_str("\"steps\":[");
        for (i, step) in self.steps.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!(
                "{{\"id\":{},\"type\":\"{:?}\",\"status\":\"{:?}\"",
                step.step_id, step.step_type, step.status
            ));
            if let Some(duration) = step.duration {
                json.push_str(&format!(
                    ",\"duration_ms\":{}",
                    duration.as_secs_f64() * 1000.0
                ));
            }
            json.push('}');
        }
        json.push_str("],");

        // Summary
        let summary = self.summary();
        json.push_str(&format!(
            "\"summary\":{{\"total\":{},\"completed\":{},\"failed\":{}}}",
            summary.total_steps, summary.completed_steps, summary.failed_steps
        ));

        json.push('}');
        json
    }
}

/// Execution summary
#[derive(Debug, Clone)]
pub struct ExecutionSummary {
    pub total_steps: usize,
    pub completed_steps: usize,
    pub failed_steps: usize,
    pub skipped_steps: usize,
    pub total_duration: Duration,
}

/// Query debugger for managing execution plans
pub struct QueryDebugger {
    plans: HashMap<String, ExecutionPlan>,
    max_plans: usize,
}

impl QueryDebugger {
    /// Create a new query debugger
    pub fn new(max_plans: usize) -> Self {
        Self {
            plans: HashMap::new(),
            max_plans,
        }
    }

    /// Start debugging a query
    pub fn start_query(
        &mut self,
        query_id: String,
        query_text: String,
        variables: HashMap<String, String>,
    ) -> String {
        // Clean up old plans if at limit
        if self.plans.len() >= self.max_plans {
            // Remove oldest plan
            if let Some(oldest_id) = self.plans.keys().next().cloned() {
                self.plans.remove(&oldest_id);
            }
        }

        let plan = ExecutionPlan::new(query_id.clone(), query_text, variables);
        self.plans.insert(query_id.clone(), plan);
        query_id
    }

    /// Add a step to a query's execution plan
    pub fn add_step(&mut self, query_id: &str, step_type: StepType) -> Option<usize> {
        self.plans
            .get_mut(query_id)
            .map(|plan| plan.add_step(step_type))
    }

    /// Start a step
    pub fn start_step(&mut self, query_id: &str, step_id: usize) {
        if let Some(plan) = self.plans.get_mut(query_id) {
            plan.start_step(step_id);
        }
    }

    /// Complete a step
    pub fn complete_step(&mut self, query_id: &str, step_id: usize, output: Option<String>) {
        if let Some(plan) = self.plans.get_mut(query_id) {
            plan.complete_step(step_id, output);
        }
    }

    /// Fail a step
    pub fn fail_step(&mut self, query_id: &str, step_id: usize, error: String) {
        if let Some(plan) = self.plans.get_mut(query_id) {
            plan.fail_step(step_id, error);
        }
    }

    /// Finalize a query execution plan
    pub fn finalize_query(&mut self, query_id: &str) {
        if let Some(plan) = self.plans.get_mut(query_id) {
            plan.finalize();
        }
    }

    /// Get an execution plan
    pub fn get_plan(&self, query_id: &str) -> Option<&ExecutionPlan> {
        self.plans.get(query_id)
    }

    /// Get all execution plans
    pub fn get_all_plans(&self) -> Vec<&ExecutionPlan> {
        self.plans.values().collect()
    }

    /// Get execution summary for a query
    pub fn get_summary(&self, query_id: &str) -> Option<ExecutionSummary> {
        self.plans.get(query_id).map(|plan| plan.summary())
    }

    /// Clear all plans
    pub fn clear(&mut self) {
        self.plans.clear();
    }

    /// Get plan count
    pub fn plan_count(&self) -> usize {
        self.plans.len()
    }

    /// Generate report for all plans
    pub fn generate_report(&self) -> String {
        let mut report = String::from("=== Query Execution Debug Report ===\n\n");
        report.push_str(&format!("Total queries tracked: {}\n\n", self.plans.len()));

        for plan in self.plans.values() {
            report.push_str(&format!("Query ID: {}\n", plan.query_id));
            report.push_str(&format!("Query: {}\n", plan.query_text));

            let summary = plan.summary();
            report.push_str(&format!(
                "Steps: {} total, {} completed, {} failed\n",
                summary.total_steps, summary.completed_steps, summary.failed_steps
            ));
            report.push_str(&format!(
                "Duration: {:.2}ms\n",
                summary.total_duration.as_secs_f64() * 1000.0
            ));
            report.push('\n');
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_plan_creation() {
        let plan = ExecutionPlan::new(
            "query-1".to_string(),
            "{ user { id name } }".to_string(),
            HashMap::new(),
        );

        assert_eq!(plan.query_id, "query-1");
        assert_eq!(plan.steps.len(), 0);
    }

    #[test]
    fn test_add_step() {
        let mut plan =
            ExecutionPlan::new("query-1".to_string(), "query".to_string(), HashMap::new());

        let step_id = plan.add_step(StepType::Parse);
        assert_eq!(step_id, 0);
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].status, StepStatus::Pending);
    }

    #[test]
    fn test_start_complete_step() {
        let mut plan =
            ExecutionPlan::new("query-1".to_string(), "query".to_string(), HashMap::new());

        let step_id = plan.add_step(StepType::Parse);
        plan.start_step(step_id);
        assert_eq!(plan.steps[step_id].status, StepStatus::Running);

        plan.complete_step(step_id, Some("parsed".to_string()));
        assert_eq!(plan.steps[step_id].status, StepStatus::Completed);
        assert!(plan.steps[step_id].duration.is_some());
    }

    #[test]
    fn test_fail_step() {
        let mut plan =
            ExecutionPlan::new("query-1".to_string(), "query".to_string(), HashMap::new());

        let step_id = plan.add_step(StepType::Validate);
        plan.start_step(step_id);
        plan.fail_step(step_id, "Validation error".to_string());

        assert_eq!(plan.steps[step_id].status, StepStatus::Failed);
        assert_eq!(
            plan.steps[step_id].error,
            Some("Validation error".to_string())
        );
    }

    #[test]
    fn test_skip_step() {
        let mut plan =
            ExecutionPlan::new("query-1".to_string(), "query".to_string(), HashMap::new());

        let step_id = plan.add_step(StepType::Optimize);
        plan.skip_step(step_id, "Optimization disabled".to_string());

        assert_eq!(plan.steps[step_id].status, StepStatus::Skipped);
        assert_eq!(
            plan.steps[step_id].metadata.get("skip_reason"),
            Some(&"Optimization disabled".to_string())
        );
    }

    #[test]
    fn test_execution_summary() {
        let mut plan =
            ExecutionPlan::new("query-1".to_string(), "query".to_string(), HashMap::new());

        let step1 = plan.add_step(StepType::Parse);
        plan.start_step(step1);
        plan.complete_step(step1, None);

        let step2 = plan.add_step(StepType::Validate);
        plan.start_step(step2);
        plan.fail_step(step2, "Error".to_string());

        let step3 = plan.add_step(StepType::Optimize);
        plan.skip_step(step3, "Skipped".to_string());

        let summary = plan.summary();
        assert_eq!(summary.total_steps, 3);
        assert_eq!(summary.completed_steps, 1);
        assert_eq!(summary.failed_steps, 1);
        assert_eq!(summary.skipped_steps, 1);
    }

    #[test]
    fn test_visualize_tree() {
        let mut plan = ExecutionPlan::new(
            "query-1".to_string(),
            "{ user { id } }".to_string(),
            HashMap::new(),
        );

        plan.add_step(StepType::Parse);
        plan.complete_step(0, None);

        plan.add_step(StepType::Validate);
        plan.complete_step(1, None);

        let tree = plan.visualize_tree();
        assert!(tree.contains("Query: query-1"));
        assert!(tree.contains("Parse"));
        assert!(tree.contains("Validate"));
    }

    #[test]
    fn test_to_json() {
        let mut plan =
            ExecutionPlan::new("query-1".to_string(), "query".to_string(), HashMap::new());

        plan.add_step(StepType::Parse);
        plan.complete_step(0, None);

        let json = plan.to_json();
        assert!(json.contains("query_id"));
        assert!(json.contains("steps"));
        assert!(json.contains("summary"));
    }

    #[test]
    fn test_query_debugger_start() {
        let mut debugger = QueryDebugger::new(100);

        let query_id =
            debugger.start_query("query-1".to_string(), "query".to_string(), HashMap::new());

        assert_eq!(query_id, "query-1");
        assert_eq!(debugger.plan_count(), 1);
    }

    #[test]
    fn test_query_debugger_add_step() {
        let mut debugger = QueryDebugger::new(100);

        debugger.start_query("query-1".to_string(), "query".to_string(), HashMap::new());
        let step_id = debugger.add_step("query-1", StepType::Parse);

        assert_eq!(step_id, Some(0));
    }

    #[test]
    fn test_query_debugger_complete_flow() {
        let mut debugger = QueryDebugger::new(100);

        debugger.start_query("query-1".to_string(), "query".to_string(), HashMap::new());

        let step_id = debugger.add_step("query-1", StepType::Parse).unwrap();
        debugger.start_step("query-1", step_id);
        debugger.complete_step("query-1", step_id, Some("parsed".to_string()));

        debugger.finalize_query("query-1");

        let plan = debugger.get_plan("query-1").unwrap();
        assert_eq!(plan.steps[0].status, StepStatus::Completed);
        assert!(plan.total_duration.is_some());
    }

    #[test]
    fn test_query_debugger_get_summary() {
        let mut debugger = QueryDebugger::new(100);

        debugger.start_query("query-1".to_string(), "query".to_string(), HashMap::new());

        let step_id = debugger.add_step("query-1", StepType::Parse).unwrap();
        debugger.complete_step("query-1", step_id, None);

        let summary = debugger.get_summary("query-1").unwrap();
        assert_eq!(summary.completed_steps, 1);
    }

    #[test]
    fn test_query_debugger_max_plans() {
        let mut debugger = QueryDebugger::new(2);

        debugger.start_query("query-1".to_string(), "query".to_string(), HashMap::new());
        debugger.start_query("query-2".to_string(), "query".to_string(), HashMap::new());
        debugger.start_query("query-3".to_string(), "query".to_string(), HashMap::new());

        // Should only keep 2 plans
        assert_eq!(debugger.plan_count(), 2);
    }

    #[test]
    fn test_query_debugger_get_all_plans() {
        let mut debugger = QueryDebugger::new(100);

        debugger.start_query("query-1".to_string(), "query".to_string(), HashMap::new());
        debugger.start_query("query-2".to_string(), "query".to_string(), HashMap::new());

        let plans = debugger.get_all_plans();
        assert_eq!(plans.len(), 2);
    }

    #[test]
    fn test_query_debugger_clear() {
        let mut debugger = QueryDebugger::new(100);

        debugger.start_query("query-1".to_string(), "query".to_string(), HashMap::new());
        assert_eq!(debugger.plan_count(), 1);

        debugger.clear();
        assert_eq!(debugger.plan_count(), 0);
    }

    #[test]
    fn test_query_debugger_generate_report() {
        let mut debugger = QueryDebugger::new(100);

        debugger.start_query(
            "query-1".to_string(),
            "{ user { id } }".to_string(),
            HashMap::new(),
        );
        debugger.add_step("query-1", StepType::Parse);

        let report = debugger.generate_report();
        assert!(report.contains("Debug Report"));
        assert!(report.contains("query-1"));
    }

    #[test]
    fn test_step_metadata() {
        let mut plan =
            ExecutionPlan::new("query-1".to_string(), "query".to_string(), HashMap::new());

        let step_id = plan.add_step(StepType::ExecuteDataSource {
            query: "SELECT * FROM users".to_string(),
            source: "postgres".to_string(),
        });

        plan.set_step_metadata(step_id, "rows_returned".to_string(), "100".to_string());

        assert_eq!(
            plan.steps[step_id].metadata.get("rows_returned"),
            Some(&"100".to_string())
        );
    }

    #[test]
    fn test_resolve_field_step() {
        let mut plan =
            ExecutionPlan::new("query-1".to_string(), "query".to_string(), HashMap::new());

        let step_id = plan.add_step(StepType::ResolveField {
            field_name: "user".to_string(),
            parent_type: "Query".to_string(),
        });

        if let StepType::ResolveField { field_name, .. } = &plan.steps[step_id].step_type {
            assert_eq!(field_name, "user");
        } else {
            panic!("Expected ResolveField step");
        }
    }

    #[test]
    fn test_finalize_sets_total_duration() {
        let mut plan =
            ExecutionPlan::new("query-1".to_string(), "query".to_string(), HashMap::new());

        assert!(plan.total_duration.is_none());

        plan.finalize();

        assert!(plan.total_duration.is_some());
    }

    #[test]
    fn test_multiple_steps_different_types() {
        let mut plan =
            ExecutionPlan::new("query-1".to_string(), "query".to_string(), HashMap::new());

        plan.add_step(StepType::Parse);
        plan.add_step(StepType::Validate);
        plan.add_step(StepType::Optimize);
        plan.add_step(StepType::Transform);
        plan.add_step(StepType::Serialize);

        assert_eq!(plan.steps.len(), 5);
        assert!(matches!(plan.steps[0].step_type, StepType::Parse));
        assert!(matches!(plan.steps[1].step_type, StepType::Validate));
        assert!(matches!(plan.steps[4].step_type, StepType::Serialize));
    }
}
