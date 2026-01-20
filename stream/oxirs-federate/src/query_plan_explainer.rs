//! # Query Plan Explainer and Visualizer
//!
//! This module provides detailed explanation and visualization of federated query execution plans,
//! helping users understand how their queries are decomposed, distributed, and executed across
//! multiple services.
//!
//! ## Features
//!
//! - Detailed plan structure breakdown
//! - Cost and performance analysis per step
//! - Service dependency visualization
//! - Optimization suggestions
//! - JSON and human-readable output formats
//!
//! ## Usage
//!
//! ```rust,ignore
//! use crate::query_plan_explainer::{QueryPlanExplainer, ExplainFormat};
//!
//! let explainer = QueryPlanExplainer::new();
//! let explanation = explainer.explain_plan(&execution_plan, ExplainFormat::Detailed)?;
//! println!("{}", explanation);
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::planner::planning::types::{ExecutionPlan, ExecutionStep, StepType};

/// Format for plan explanation output
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplainFormat {
    /// Brief summary
    Brief,
    /// Detailed step-by-step explanation
    Detailed,
    /// JSON format for programmatic access
    Json,
    /// Tree structure visualization
    Tree,
    /// Cost analysis focused
    CostAnalysis,
}

/// Query plan explainer
pub struct QueryPlanExplainer {
    config: ExplainerConfig,
}

/// Configuration for plan explanation
#[derive(Debug, Clone)]
pub struct ExplainerConfig {
    /// Include cost estimates
    pub include_costs: bool,
    /// Include service details
    pub include_services: bool,
    /// Include optimization suggestions
    pub include_suggestions: bool,
    /// Maximum depth for tree visualization
    pub max_tree_depth: usize,
    /// Show timing estimates
    pub show_timing_estimates: bool,
}

impl Default for ExplainerConfig {
    fn default() -> Self {
        Self {
            include_costs: true,
            include_services: true,
            include_suggestions: true,
            max_tree_depth: 10,
            show_timing_estimates: true,
        }
    }
}

/// Explanation of an execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanExplanation {
    /// Summary of the plan
    pub summary: String,
    /// Total estimated cost
    pub total_cost: u64,
    /// Number of steps
    pub step_count: usize,
    /// Number of services involved
    pub service_count: usize,
    /// Steps breakdown
    pub steps: Vec<StepExplanation>,
    /// Service dependencies
    pub service_dependencies: HashMap<String, Vec<String>>,
    /// Parallelization opportunities (step IDs)
    pub parallel_steps: Vec<Vec<String>>,
    /// Optimization suggestions
    pub suggestions: Vec<OptimizationSuggestion>,
    /// Critical path (longest execution sequence) (step IDs)
    pub critical_path: Vec<String>,
    /// Estimated total execution time (milliseconds)
    pub estimated_duration_ms: u64,
}

/// Explanation of a single execution step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepExplanation {
    /// Step ID
    pub step_id: String,
    /// Step type description
    pub step_type: String,
    /// Service ID if applicable
    pub service_id: Option<String>,
    /// Query fragment or operation
    pub operation: String,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Estimated duration (milliseconds)
    pub estimated_duration_ms: u64,
    /// Dependencies on other steps (step IDs)
    pub dependencies: Vec<String>,
    /// Whether this step can run in parallel
    pub can_parallelize: bool,
    /// Detailed description
    pub description: String,
}

/// Optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    /// Suggestion category
    pub category: SuggestionCategory,
    /// Severity level
    pub severity: SuggestionSeverity,
    /// Description of the suggestion
    pub description: String,
    /// Steps affected (step IDs)
    pub affected_steps: Vec<String>,
    /// Potential improvement
    pub potential_improvement: String,
}

/// Category of optimization suggestion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionCategory {
    /// Join ordering optimization
    JoinOrdering,
    /// Service selection
    ServiceSelection,
    /// Caching opportunity
    Caching,
    /// Parallelization
    Parallelization,
    /// Filter pushdown
    FilterPushdown,
    /// Data transfer reduction
    DataTransfer,
    /// Index usage
    IndexUsage,
}

/// Severity of optimization suggestion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionSeverity {
    /// Low impact
    Low,
    /// Medium impact
    Medium,
    /// High impact
    High,
    /// Critical impact
    Critical,
}

impl QueryPlanExplainer {
    /// Create a new query plan explainer with default configuration
    pub fn new() -> Self {
        Self {
            config: ExplainerConfig::default(),
        }
    }

    /// Create a new query plan explainer with custom configuration
    pub fn with_config(config: ExplainerConfig) -> Self {
        Self { config }
    }

    /// Explain an execution plan
    pub fn explain_plan(&self, plan: &ExecutionPlan, format: ExplainFormat) -> Result<String> {
        let explanation = self.analyze_plan(plan)?;

        match format {
            ExplainFormat::Brief => Ok(self.format_brief(&explanation)),
            ExplainFormat::Detailed => Ok(self.format_detailed(&explanation)),
            ExplainFormat::Json => Ok(serde_json::to_string_pretty(&explanation)?),
            ExplainFormat::Tree => Ok(self.format_tree(&explanation)),
            ExplainFormat::CostAnalysis => Ok(self.format_cost_analysis(&explanation)),
        }
    }

    /// Analyze an execution plan and generate explanation
    fn analyze_plan(&self, plan: &ExecutionPlan) -> Result<PlanExplanation> {
        let steps = plan
            .steps
            .iter()
            .map(|step| self.explain_step(step, plan))
            .collect::<Vec<_>>();

        let service_count = steps
            .iter()
            .filter_map(|s| s.service_id.as_ref())
            .collect::<HashSet<_>>()
            .len();

        let service_dependencies = self.analyze_service_dependencies(&steps);
        let parallel_steps = self.identify_parallel_steps(&steps);
        let critical_path = self.compute_critical_path(&steps);
        let suggestions = if self.config.include_suggestions {
            self.generate_suggestions(plan, &steps)
        } else {
            Vec::new()
        };

        let total_cost = steps.iter().map(|s| s.estimated_cost as u64).sum();
        let estimated_duration_ms = self.estimate_total_duration(&steps, &critical_path);

        let summary = self.generate_summary(steps.len(), service_count, &parallel_steps);

        Ok(PlanExplanation {
            summary,
            total_cost,
            step_count: steps.len(),
            service_count,
            steps,
            service_dependencies,
            parallel_steps,
            suggestions,
            critical_path,
            estimated_duration_ms,
        })
    }

    /// Explain a single execution step
    fn explain_step(&self, step: &ExecutionStep, _plan: &ExecutionPlan) -> StepExplanation {
        let step_id = step.step_id.clone();
        let step_type = format!("{:?}", step.step_type);
        let service_id = step.service_id.clone();

        let operation = match &step.step_type {
            StepType::ServiceQuery => {
                format!("Execute query on service: {}", step.query_fragment)
            }
            StepType::GraphQLQuery => {
                format!("Execute GraphQL query: {}", step.query_fragment)
            }
            StepType::Join => "Join results from previous steps".to_string(),
            StepType::Filter => "Apply filters".to_string(),
            StepType::Aggregate => "Aggregate results".to_string(),
            StepType::Sort => "Sort results".to_string(),
            StepType::EntityResolution => "Resolve entities across services".to_string(),
            StepType::ResultStitching => "Stitch results together".to_string(),
            StepType::SchemaStitch => "Stitch schemas together".to_string(),
            StepType::Union => "Union results from multiple sources".to_string(),
        };

        let estimated_cost = step.estimated_cost;
        let estimated_duration_ms = self.estimate_step_duration(step);

        let dependencies = step.dependencies.clone();
        let can_parallelize = dependencies.is_empty() || dependencies.len() <= 1;

        let description = self.generate_step_description(step);

        StepExplanation {
            step_id,
            step_type,
            service_id,
            operation,
            estimated_cost,
            estimated_duration_ms,
            dependencies,
            can_parallelize,
            description,
        }
    }

    /// Generate detailed description for a step
    fn generate_step_description(&self, step: &ExecutionStep) -> String {
        let mut desc = String::new();

        match &step.step_type {
            StepType::ServiceQuery | StepType::GraphQLQuery => {
                if let Some(service_id) = &step.service_id {
                    desc.push_str(&format!("Query service '{}' ", service_id));
                }
                if step.query_fragment.len() > 100 {
                    desc.push_str(&format!(
                        "with query ({}... characters)",
                        step.query_fragment.len()
                    ));
                } else {
                    desc.push_str(&format!("with query: {}", step.query_fragment));
                }
            }
            StepType::Join => {
                desc.push_str("Perform join operation ");
                if !step.dependencies.is_empty() {
                    desc.push_str(&format!(
                        "combining results from steps {:?}",
                        step.dependencies
                    ));
                }
            }
            StepType::Filter => {
                desc.push_str("Apply filtering conditions to reduce result set");
            }
            StepType::Aggregate => {
                desc.push_str("Aggregate data (COUNT, SUM, AVG, etc.)");
            }
            StepType::Sort => {
                desc.push_str("Sort results by specified criteria");
            }
            StepType::EntityResolution => {
                desc.push_str("Resolve entity references across different services");
            }
            StepType::ResultStitching => {
                desc.push_str("Combine and stitch results from multiple sources");
            }
            StepType::SchemaStitch => {
                desc.push_str("Stitch schemas together for federation");
            }
            StepType::Union => {
                desc.push_str("Union results from multiple query branches");
            }
        }

        // Note: estimated_result_size not available on ExecutionStep
        // Could be added in future enhancement

        desc
    }

    /// Estimate duration for a step
    fn estimate_step_duration(&self, step: &ExecutionStep) -> u64 {
        if !self.config.show_timing_estimates {
            return 0;
        }

        // Base duration estimates (in milliseconds)
        let base_duration = match &step.step_type {
            StepType::ServiceQuery => 50, // Network call
            StepType::GraphQLQuery => 40,
            StepType::Join => 10,
            StepType::Filter => 2,
            StepType::Aggregate => 5,
            StepType::Sort => 8,
            StepType::EntityResolution => 15,
            StepType::ResultStitching => 5,
            StepType::SchemaStitch => 8,
            StepType::Union => 3,
        };

        // Use estimated cost as a size proxy
        let cost_multiplier = 1.0 + (step.estimated_cost / 100.0).min(10.0);

        (base_duration as f64 * cost_multiplier) as u64
    }

    /// Analyze service dependencies
    fn analyze_service_dependencies(
        &self,
        steps: &[StepExplanation],
    ) -> HashMap<String, Vec<String>> {
        let mut dependencies: HashMap<String, HashSet<String>> = HashMap::new();

        // Create step ID to step map
        let step_map: HashMap<&String, &StepExplanation> =
            steps.iter().map(|s| (&s.step_id, s)).collect();

        for step in steps {
            if let Some(service_id) = &step.service_id {
                // Find services this step depends on
                for dep_id in &step.dependencies {
                    if let Some(dep_step) = step_map.get(dep_id) {
                        if let Some(dep_service) = &dep_step.service_id {
                            if dep_service != service_id {
                                dependencies
                                    .entry(service_id.clone())
                                    .or_default()
                                    .insert(dep_service.clone());
                            }
                        }
                    }
                }
            }
        }

        dependencies
            .into_iter()
            .map(|(k, v)| (k, v.into_iter().collect()))
            .collect()
    }

    /// Identify steps that can run in parallel
    fn identify_parallel_steps(&self, steps: &[StepExplanation]) -> Vec<Vec<String>> {
        let mut parallel_groups = Vec::new();
        let mut processed = HashSet::new();

        for step in steps {
            if processed.contains(&step.step_id) {
                continue;
            }

            if step.dependencies.is_empty() {
                // This step has no dependencies, find others like it
                let mut group = vec![step.step_id.clone()];
                processed.insert(step.step_id.clone());

                for other_step in steps {
                    if other_step.step_id != step.step_id
                        && !processed.contains(&other_step.step_id)
                        && other_step.dependencies.is_empty()
                    {
                        group.push(other_step.step_id.clone());
                        processed.insert(other_step.step_id.clone());
                    }
                }

                if group.len() > 1 {
                    parallel_groups.push(group);
                }
            }
        }

        parallel_groups
    }

    /// Compute the critical path (longest execution sequence)
    fn compute_critical_path(&self, steps: &[StepExplanation]) -> Vec<String> {
        if steps.is_empty() {
            return Vec::new();
        }

        // Simple heuristic: follow the path with highest cumulative cost
        let mut path = Vec::new();
        let current_id = steps.first().map(|s| s.step_id.clone());
        if current_id.is_none() {
            return path;
        }

        let mut current_id = current_id.unwrap();
        let mut visited = HashSet::new();

        path.push(current_id.clone());
        visited.insert(current_id.clone());

        // Find steps that depend on current step
        loop {
            let mut next_step: Option<&StepExplanation> = None;
            let mut max_cost = 0.0;

            for step in steps {
                if !visited.contains(&step.step_id)
                    && step.dependencies.contains(&current_id)
                    && step.estimated_cost > max_cost
                {
                    max_cost = step.estimated_cost;
                    next_step = Some(step);
                }
            }

            if let Some(step) = next_step {
                path.push(step.step_id.clone());
                visited.insert(step.step_id.clone());
                current_id = step.step_id.clone();
            } else {
                break;
            }
        }

        path
    }

    /// Estimate total execution duration
    fn estimate_total_duration(&self, steps: &[StepExplanation], critical_path: &[String]) -> u64 {
        let step_map: HashMap<&String, &StepExplanation> =
            steps.iter().map(|s| (&s.step_id, s)).collect();

        critical_path
            .iter()
            .filter_map(|id| step_map.get(id))
            .map(|step| step.estimated_duration_ms)
            .sum()
    }

    /// Generate summary of the plan
    fn generate_summary(
        &self,
        step_count: usize,
        service_count: usize,
        parallel_steps: &[Vec<String>],
    ) -> String {
        format!(
            "Federated query execution plan with {} steps across {} services. {} parallelization opportunities identified.",
            step_count,
            service_count,
            parallel_steps.len()
        )
    }

    /// Generate optimization suggestions
    fn generate_suggestions(
        &self,
        _plan: &ExecutionPlan,
        steps: &[StepExplanation],
    ) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        // Check for sequential service queries that could be parallelized
        let sequential_queries: Vec<String> = steps
            .iter()
            .filter(|s| matches!(s.step_type.as_str(), "ServiceQuery" | "GraphQLQuery"))
            .map(|s| s.step_id.clone())
            .collect();

        if sequential_queries.len() > 1 {
            suggestions.push(OptimizationSuggestion {
                category: SuggestionCategory::Parallelization,
                severity: SuggestionSeverity::High,
                description: format!(
                    "Found {} service queries that could potentially run in parallel",
                    sequential_queries.len()
                ),
                affected_steps: sequential_queries,
                potential_improvement: "Reduce total execution time by up to 50%".to_string(),
            });
        }

        // Check for high-cost steps that might benefit from caching
        for step in steps {
            if step.estimated_cost > 100.0 {
                suggestions.push(OptimizationSuggestion {
                    category: SuggestionCategory::Caching,
                    severity: SuggestionSeverity::Medium,
                    description: format!(
                        "Step {} has high cost ({:.0}) - consider caching",
                        step.step_id, step.estimated_cost
                    ),
                    affected_steps: vec![step.step_id.clone()],
                    potential_improvement: "Reduce repeated query overhead".to_string(),
                });
            }
        }

        // Check for joins that could be optimized
        let join_steps: Vec<_> = steps.iter().filter(|s| s.step_type == "Join").collect();

        if join_steps.len() > 2 {
            suggestions.push(OptimizationSuggestion {
                category: SuggestionCategory::JoinOrdering,
                severity: SuggestionSeverity::High,
                description: "Multiple joins detected - verify join ordering is optimal"
                    .to_string(),
                affected_steps: join_steps.iter().map(|s| s.step_id.clone()).collect(),
                potential_improvement: "Optimize join order to process smaller result sets first"
                    .to_string(),
            });
        }

        suggestions
    }

    /// Format brief explanation
    fn format_brief(&self, explanation: &PlanExplanation) -> String {
        format!(
            "{}\n\nTotal Cost: {}\nSteps: {}\nServices: {}\nEstimated Duration: {}ms\nParallel Opportunities: {}",
            explanation.summary,
            explanation.total_cost,
            explanation.step_count,
            explanation.service_count,
            explanation.estimated_duration_ms,
            explanation.parallel_steps.len()
        )
    }

    /// Format detailed explanation
    fn format_detailed(&self, explanation: &PlanExplanation) -> String {
        let mut output = String::new();

        output.push_str("=== QUERY EXECUTION PLAN ===\n\n");
        output.push_str(&format!("{}\n\n", explanation.summary));

        output.push_str(&format!(
            "Total Estimated Cost: {}\n",
            explanation.total_cost
        ));
        output.push_str(&format!(
            "Total Estimated Duration: {}ms\n",
            explanation.estimated_duration_ms
        ));
        output.push_str(&format!("Number of Steps: {}\n", explanation.step_count));
        output.push_str(&format!(
            "Services Involved: {}\n\n",
            explanation.service_count
        ));

        output.push_str("--- EXECUTION STEPS ---\n\n");
        for step in &explanation.steps {
            output.push_str(&format!("Step {}: {}\n", step.step_id, step.step_type));
            output.push_str(&format!("  {}\n", step.description));
            if let Some(service_id) = &step.service_id {
                output.push_str(&format!("  Service: {}\n", service_id));
            }
            output.push_str(&format!("  Cost: {:.2}\n", step.estimated_cost));
            output.push_str(&format!("  Duration: {}ms\n", step.estimated_duration_ms));
            if !step.dependencies.is_empty() {
                output.push_str(&format!("  Depends on steps: {:?}\n", step.dependencies));
            }
            if step.can_parallelize {
                output.push_str("  ✓ Can run in parallel\n");
            }
            output.push('\n');
        }

        if !explanation.parallel_steps.is_empty() {
            output.push_str("--- PARALLELIZATION OPPORTUNITIES ---\n\n");
            for (idx, group) in explanation.parallel_steps.iter().enumerate() {
                output.push_str(&format!("Parallel Group {}: Steps {:?}\n", idx + 1, group));
            }
            output.push('\n');
        }

        if !explanation.critical_path.is_empty() {
            output.push_str("--- CRITICAL PATH ---\n\n");
            output.push_str(&format!("Steps: {:?}\n", explanation.critical_path));
            output.push_str(&format!(
                "Total Duration: {}ms\n\n",
                explanation.estimated_duration_ms
            ));
        }

        if !explanation.suggestions.is_empty() {
            output.push_str("--- OPTIMIZATION SUGGESTIONS ---\n\n");
            for (idx, suggestion) in explanation.suggestions.iter().enumerate() {
                output.push_str(&format!(
                    "{}. [{:?}] {:?}: {}\n",
                    idx + 1,
                    suggestion.severity,
                    suggestion.category,
                    suggestion.description
                ));
                output.push_str(&format!(
                    "   Potential: {}\n",
                    suggestion.potential_improvement
                ));
                output.push_str(&format!(
                    "   Affects steps: {:?}\n\n",
                    suggestion.affected_steps
                ));
            }
        }

        output
    }

    /// Format tree structure
    fn format_tree(&self, explanation: &PlanExplanation) -> String {
        let mut output = String::new();
        output.push_str("Query Execution Tree:\n\n");

        let mut printed = HashSet::new();

        // Start with steps that have no dependencies
        for step in &explanation.steps {
            if step.dependencies.is_empty() && !printed.contains(&step.step_id) {
                self.format_tree_recursive(
                    &explanation.steps,
                    &step.step_id,
                    "",
                    &mut output,
                    &mut printed,
                    0,
                );
            }
        }

        output
    }

    /// Recursively format tree structure
    fn format_tree_recursive(
        &self,
        steps: &[StepExplanation],
        step_id: &str,
        prefix: &str,
        output: &mut String,
        printed: &mut HashSet<String>,
        depth: usize,
    ) {
        if depth >= self.config.max_tree_depth || printed.contains(step_id) {
            return;
        }

        if let Some(step) = steps.iter().find(|s| s.step_id == step_id) {
            printed.insert(step_id.to_string());

            output.push_str(&format!(
                "{}├─ [{}] {} (cost: {:.2}, {}ms)\n",
                prefix,
                step.step_id,
                step.step_type,
                step.estimated_cost,
                step.estimated_duration_ms
            ));

            if let Some(service_id) = &step.service_id {
                output.push_str(&format!("{}│  Service: {}\n", prefix, service_id));
            }

            // Find steps that depend on this one
            let dependents: Vec<String> = steps
                .iter()
                .filter(|s| s.dependencies.contains(&step.step_id))
                .map(|s| s.step_id.clone())
                .collect();

            for (idx, dependent) in dependents.iter().enumerate() {
                let is_last = idx == dependents.len() - 1;
                let new_prefix = if is_last {
                    format!("{}   ", prefix)
                } else {
                    format!("{}│  ", prefix)
                };
                self.format_tree_recursive(
                    steps,
                    dependent,
                    &new_prefix,
                    output,
                    printed,
                    depth + 1,
                );
            }
        }
    }

    /// Format cost analysis
    fn format_cost_analysis(&self, explanation: &PlanExplanation) -> String {
        let mut output = String::new();

        output.push_str("=== COST ANALYSIS ===\n\n");
        output.push_str(&format!(
            "Total Estimated Cost: {}\n\n",
            explanation.total_cost
        ));

        // Group by step type
        let mut cost_by_type: HashMap<String, f64> = HashMap::new();
        for step in &explanation.steps {
            *cost_by_type.entry(step.step_type.clone()).or_insert(0.0) += step.estimated_cost;
        }

        output.push_str("Cost Breakdown by Operation Type:\n");
        let mut sorted_types: Vec<_> = cost_by_type.iter().collect();
        sorted_types.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (step_type, cost) in sorted_types {
            let percentage = (*cost / explanation.total_cost as f64) * 100.0;
            output.push_str(&format!(
                "  {}: {:.2} ({:.1}%)\n",
                step_type, cost, percentage
            ));
        }

        output.push('\n');

        // Group by service
        let mut cost_by_service: HashMap<String, f64> = HashMap::new();
        for step in &explanation.steps {
            if let Some(service_id) = &step.service_id {
                *cost_by_service.entry(service_id.clone()).or_insert(0.0) += step.estimated_cost;
            }
        }

        if !cost_by_service.is_empty() {
            output.push_str("Cost Breakdown by Service:\n");
            let mut sorted_services: Vec<_> = cost_by_service.iter().collect();
            sorted_services
                .sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (service_id, cost) in sorted_services {
                let percentage = (*cost / explanation.total_cost as f64) * 100.0;
                output.push_str(&format!(
                    "  {}: {:.2} ({:.1}%)\n",
                    service_id, cost, percentage
                ));
            }
        }

        output
    }
}

impl Default for QueryPlanExplainer {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SuggestionSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::planning::types::ExecutionStep;
    use crate::planner::planning::types::StepType;

    fn create_test_plan() -> ExecutionPlan {
        use std::collections::HashMap;
        use std::time::Duration;

        ExecutionPlan {
            query_id: "test_query_1".to_string(),
            steps: vec![
                ExecutionStep {
                    step_id: "step_0".to_string(),
                    step_type: StepType::ServiceQuery,
                    service_id: Some("service1".to_string()),
                    service_url: Some("http://service1.example.com/sparql".to_string()),
                    auth_config: None,
                    query_fragment: "SELECT ?s WHERE { ?s rdf:type foaf:Person }".to_string(),
                    dependencies: vec![],
                    estimated_cost: 100.0,
                    timeout: Duration::from_secs(30),
                    retry_config: None,
                },
                ExecutionStep {
                    step_id: "step_1".to_string(),
                    step_type: StepType::ServiceQuery,
                    service_id: Some("service2".to_string()),
                    service_url: Some("http://service2.example.com/sparql".to_string()),
                    auth_config: None,
                    query_fragment: "SELECT ?name WHERE { ?s foaf:name ?name }".to_string(),
                    dependencies: vec![],
                    estimated_cost: 80.0,
                    timeout: Duration::from_secs(30),
                    retry_config: None,
                },
                ExecutionStep {
                    step_id: "step_2".to_string(),
                    step_type: StepType::Join,
                    service_id: None,
                    service_url: None,
                    auth_config: None,
                    query_fragment: String::new(),
                    dependencies: vec!["step_0".to_string(), "step_1".to_string()],
                    estimated_cost: 50.0,
                    timeout: Duration::from_secs(30),
                    retry_config: None,
                },
            ],
            estimated_total_cost: 230.0,
            max_parallelism: 2,
            planning_time: Duration::from_millis(100),
            cache_key: Some("test_cache_key".to_string()),
            metadata: HashMap::new(),
            parallelizable_steps: vec![vec!["step_0".to_string(), "step_1".to_string()]],
        }
    }

    #[test]
    fn test_explainer_creation() {
        let explainer = QueryPlanExplainer::new();
        assert!(explainer.config.include_costs);
        assert!(explainer.config.include_services);
    }

    #[test]
    fn test_explain_plan_brief() {
        let explainer = QueryPlanExplainer::new();
        let plan = create_test_plan();

        let explanation = explainer.explain_plan(&plan, ExplainFormat::Brief);
        assert!(explanation.is_ok());

        let output = explanation.unwrap();
        assert!(output.contains("Total Cost"));
        assert!(output.contains("Steps: 3"));
        assert!(output.contains("Services: 2"));
    }

    #[test]
    fn test_explain_plan_detailed() {
        let explainer = QueryPlanExplainer::new();
        let plan = create_test_plan();

        let explanation = explainer.explain_plan(&plan, ExplainFormat::Detailed);
        assert!(explanation.is_ok());

        let output = explanation.unwrap();
        assert!(output.contains("EXECUTION STEPS"));
        assert!(output.contains("ServiceQuery"));
        assert!(output.contains("Join"));
    }

    #[test]
    fn test_explain_plan_json() {
        let explainer = QueryPlanExplainer::new();
        let plan = create_test_plan();

        let explanation = explainer.explain_plan(&plan, ExplainFormat::Json);
        assert!(explanation.is_ok());

        let output = explanation.unwrap();
        let parsed: Result<PlanExplanation, _> = serde_json::from_str(&output);
        assert!(parsed.is_ok());
    }

    #[test]
    fn test_explain_plan_tree() {
        let explainer = QueryPlanExplainer::new();
        let plan = create_test_plan();

        let explanation = explainer.explain_plan(&plan, ExplainFormat::Tree);
        assert!(explanation.is_ok());

        let output = explanation.unwrap();
        assert!(output.contains("Query Execution Tree"));
        assert!(output.contains("├─"));
    }

    #[test]
    fn test_explain_plan_cost_analysis() {
        let explainer = QueryPlanExplainer::new();
        let plan = create_test_plan();

        let explanation = explainer.explain_plan(&plan, ExplainFormat::CostAnalysis);
        assert!(explanation.is_ok());

        let output = explanation.unwrap();
        assert!(output.contains("COST ANALYSIS"));
        assert!(output.contains("Cost Breakdown"));
    }

    #[test]
    fn test_parallel_step_identification() {
        let explainer = QueryPlanExplainer::new();
        let plan = create_test_plan();

        let explanation = explainer.analyze_plan(&plan).unwrap();
        assert!(!explanation.parallel_steps.is_empty());
        assert_eq!(
            explanation.parallel_steps[0],
            vec!["step_0".to_string(), "step_1".to_string()]
        );
    }

    #[test]
    fn test_optimization_suggestions() {
        let explainer = QueryPlanExplainer::new();
        let plan = create_test_plan();

        let explanation = explainer.analyze_plan(&plan).unwrap();
        assert!(!explanation.suggestions.is_empty());

        // Should suggest parallelization
        let has_parallel_suggestion = explanation
            .suggestions
            .iter()
            .any(|s| s.category == SuggestionCategory::Parallelization);
        assert!(has_parallel_suggestion);
    }

    #[test]
    fn test_critical_path() {
        let explainer = QueryPlanExplainer::new();
        let plan = create_test_plan();

        let explanation = explainer.analyze_plan(&plan).unwrap();
        assert!(!explanation.critical_path.is_empty());

        // Critical path should contain the join step
        assert!(explanation.critical_path.contains(&"step_2".to_string()));
    }

    #[test]
    fn test_service_dependencies() {
        let explainer = QueryPlanExplainer::new();
        let plan = create_test_plan();

        let explanation = explainer.analyze_plan(&plan).unwrap();
        // Two services, no cross-service dependencies in this simple case
        assert_eq!(explanation.service_count, 2);
    }

    #[test]
    fn test_estimated_duration() {
        let explainer = QueryPlanExplainer::new();
        let plan = create_test_plan();

        let explanation = explainer.analyze_plan(&plan).unwrap();
        assert!(explanation.estimated_duration_ms > 0);
    }
}
