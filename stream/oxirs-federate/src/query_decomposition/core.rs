//! Core query decomposition implementation
//!
//! This module contains the main QueryDecomposer implementation with the primary
//! decompose method and core orchestration logic.

use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::{
    planner::{ExecutionPlan, ExecutionStep, QueryInfo, StepType},
    FederatedService, ServiceCapability, ServiceRegistry,
};

use super::{
    graph_analysis::*,
    plan_generation::*,
    cost_estimation::*,
    pattern_analysis::*,
    types::*,
};

impl QueryDecomposer {
    /// Create a new query decomposer
    pub fn new() -> Self {
        Self {
            config: DecomposerConfig::default(),
            cost_estimator: CostEstimator::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: DecomposerConfig) -> Self {
        Self {
            config,
            cost_estimator: CostEstimator::new(),
        }
    }

    /// Decompose a query into an optimized execution plan
    pub async fn decompose(
        &self,
        query_info: &QueryInfo,
        registry: &ServiceRegistry,
    ) -> Result<DecompositionResult> {
        let start_time = Instant::now();
        
        info!(
            "Decomposing query with {} patterns",
            query_info.patterns.len()
        );

        // Build query graph representation
        let query_graph = self.build_query_graph(query_info)?;

        // Find connected components (subqueries that can be executed independently)
        let components = self.find_connected_components(&query_graph);

        // Generate candidate plans for each component
        let mut component_plans = Vec::new();
        for component in components {
            let plans = self.generate_component_plans(&component, registry)?;
            component_plans.push(plans);
        }

        // Select optimal plan combination
        let optimal_combination = self.select_optimal_plan_combination(&component_plans)?;

        // Build final execution plan with proper ordering
        let execution_plan = self.build_execution_plan(optimal_combination, query_info)?;

        let decomposition_time = start_time.elapsed();

        Ok(DecompositionResult {
            plan: execution_plan,
            statistics: self.calculate_decomposition_stats(&component_plans, decomposition_time),
        })
    }

    /// Select optimal combination of component plans
    pub fn select_optimal_plan_combination(
        &self,
        component_plans: &[Vec<ComponentPlan>],
    ) -> Result<Vec<ComponentPlan>> {
        let mut optimal_combination = Vec::new();

        for plans in component_plans {
            if plans.is_empty() {
                return Err(anyhow!("No plans available for component"));
            }

            // Select plan based on optimization strategy
            let selected = match self.config.optimization_strategy {
                OptimizationStrategy::MinimizeCost => {
                    plans.iter().min_by_key(|p| p.total_cost as u64).unwrap()
                }
                OptimizationStrategy::MinimizeTime => plans
                    .iter()
                    .min_by_key(|p| {
                        if p.requires_join {
                            p.steps.len() * 2
                        } else {
                            p.steps.len()
                        }
                    })
                    .unwrap(),
                OptimizationStrategy::MinimizeTransfer => plans
                    .iter()
                    .min_by_key(|p| p.steps.iter().map(|s| s.estimated_results).sum::<u64>())
                    .unwrap(),
                OptimizationStrategy::Balanced => {
                    // Balance between cost, time, and data transfer
                    plans
                        .iter()
                        .min_by_key(|p| {
                            let cost_factor = p.total_cost as u64;
                            let time_factor = p.steps.len() as u64 * 100;
                            let transfer_factor =
                                p.steps.iter().map(|s| s.estimated_results).sum::<u64>() / 1000;
                            cost_factor + time_factor + transfer_factor
                        })
                        .unwrap()
                }
            };

            optimal_combination.push(selected.clone());
        }

        Ok(optimal_combination)
    }

    /// Build final execution plan from component plans
    pub fn build_execution_plan(
        &self,
        component_plans: Vec<ComponentPlan>,
        query_info: &QueryInfo,
    ) -> Result<ExecutionPlan> {
        let mut plan = ExecutionPlan {
            query_id: uuid::Uuid::new_v4().to_string(),
            steps: Vec::new(),
            estimated_total_cost: 0.0,
            max_parallelism: 1,
            planning_time: Duration::from_millis(0),
            cache_key: None,
            metadata: HashMap::new(),
            parallelizable_steps: Vec::new(),
        };

        let mut step_ids_by_component = Vec::new();

        // Create execution steps for each component
        for (comp_idx, comp_plan) in component_plans.iter().enumerate() {
            let mut component_step_ids = Vec::new();

            for (step_idx, plan_step) in comp_plan.steps.iter().enumerate() {
                let step = ExecutionStep {
                    step_id: uuid::Uuid::new_v4().to_string(),
                    step_type: StepType::ServiceQuery,
                    service_id: Some(plan_step.service_id.clone()),
                    query_fragment: self.build_step_query(plan_step)?,
                    dependencies: Vec::new(),
                    estimated_cost: plan_step.estimated_cost,
                    timeout: Duration::from_millis((plan_step.estimated_cost * 10.0) as u64),
                    retry_config: None,
                };

                component_step_ids.push(step.step_id.clone());
                plan.steps.push(step);
            }

            // Add join step if component requires it
            if comp_plan.requires_join && comp_plan.steps.len() > 1 {
                let join_step = ExecutionStep {
                    step_id: uuid::Uuid::new_v4().to_string(),
                    step_type: StepType::Join,
                    service_id: None,
                    query_fragment: format!("-- Join {} results", comp_plan.steps.len()),
                    dependencies: component_step_ids.clone(),
                    estimated_cost: 10.0,
                    timeout: Duration::from_millis(50),
                    retry_config: None,
                };

                let join_id = join_step.step_id.clone();
                plan.steps.push(join_step);
                step_ids_by_component.push(vec![join_id]);
            } else {
                step_ids_by_component.push(component_step_ids);
            }
        }

        // Set parallelizable steps (component groups can run in parallel)
        plan.parallelizable_steps = step_ids_by_component;

        // Calculate total estimated duration
        let max_component_duration = component_plans
            .iter()
            .map(|cp| cp.steps.len() as u64 * 100) // Simple duration estimation
            .max()
            .unwrap_or(100);
        plan.planning_time = Duration::from_millis(max_component_duration);

        Ok(plan)
    }

    /// Build SPARQL query for a plan step
    pub fn build_step_query(&self, step: &PlanStep) -> Result<String> {
        let mut query_parts = Vec::new();
        query_parts.push("SELECT *".to_string());
        query_parts.push("WHERE {".to_string());

        // Add patterns
        for (_, pattern) in &step.patterns {
            query_parts.push(format!(
                "  {} {} {} .",
                pattern.subject.as_ref().unwrap_or(&"?s".to_string()),
                pattern.predicate.as_ref().unwrap_or(&"?p".to_string()),
                pattern.object.as_ref().unwrap_or(&"?o".to_string())
            ));
        }

        // Add filters
        for filter in &step.filters {
            query_parts.push(format!("  FILTER({})", filter.expression));
        }

        query_parts.push("}".to_string());

        Ok(query_parts.join("\n"))
    }

    /// Extract variables from a plan step
    pub fn extract_step_variables(&self, step: &PlanStep) -> HashSet<String> {
        let mut variables = HashSet::new();

        for (_, pattern) in &step.patterns {
            if let Some(ref subject) = pattern.subject {
                if subject.starts_with('?') {
                    variables.insert(subject.clone());
                }
            }
            if let Some(ref predicate) = pattern.predicate {
                if predicate.starts_with('?') {
                    variables.insert(predicate.clone());
                }
            }
            if let Some(ref object) = pattern.object {
                if object.starts_with('?') {
                    variables.insert(object.clone());
                }
            }
        }

        variables
    }

    /// Calculate decomposition statistics
    pub fn calculate_decomposition_stats(
        &self,
        component_plans: &[Vec<ComponentPlan>],
        decomposition_time: Duration,
    ) -> DecompositionStatistics {
        let total_patterns: usize = component_plans
            .iter()
            .flat_map(|plans| plans.iter())
            .map(|plan| plan.steps.iter().map(|step| step.patterns.len()).sum::<usize>())
            .max()
            .unwrap_or(0);

        let plans_evaluated: usize = component_plans.iter().map(|plans| plans.len()).sum();

        let selected_strategy = component_plans
            .iter()
            .flat_map(|plans| plans.iter())
            .next()
            .map(|plan| plan.strategy)
            .unwrap_or(PlanStrategy::SingleService);

        let estimated_total_cost: f64 = component_plans
            .iter()
            .flat_map(|plans| plans.iter())
            .next()
            .map(|plan| plan.total_cost)
            .unwrap_or(0.0);

        DecompositionStatistics {
            total_patterns,
            components_found: component_plans.len(),
            plans_evaluated,
            selected_strategy,
            estimated_total_cost,
            decomposition_time,
        }
    }
}

impl Default for QueryDecomposer {
    fn default() -> Self {
        Self::new()
    }
}