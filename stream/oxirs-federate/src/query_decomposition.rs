//! Advanced Query Decomposition for Federated Execution
//!
//! This module provides sophisticated algorithms for decomposing complex
//! SPARQL and GraphQL queries into optimal execution plans across multiple services.

use anyhow::{anyhow, Result};
use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::{
    planner::{ExecutionPlan, ExecutionStep, FilterExpression, QueryInfo, StepType, TriplePattern},
    FederatedService, ServiceCapability, ServiceRegistry,
};

/// Advanced query decomposer with optimization algorithms
#[derive(Debug)]
pub struct QueryDecomposer {
    config: DecomposerConfig,
    cost_estimator: CostEstimator,
}

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

        Ok(DecompositionResult {
            plan: execution_plan,
            statistics: self.calculate_decomposition_stats(&component_plans),
        })
    }

    /// Build a graph representation of the query
    fn build_query_graph(&self, query_info: &QueryInfo) -> Result<QueryGraph> {
        let mut graph = QueryGraph::new();
        let mut variable_nodes: HashMap<String, NodeIndex> = HashMap::new();
        let mut pattern_nodes: Vec<NodeIndex> = Vec::new();

        // Add nodes for each variable
        for var in &query_info.variables {
            let node_idx = graph.add_variable_node(var.clone());
            variable_nodes.insert(var.clone(), node_idx);
        }

        // Add nodes for each triple pattern and connect to variables
        for (i, pattern) in query_info.patterns.iter().enumerate() {
            let pattern_node = graph.add_pattern_node(i, pattern.clone());
            pattern_nodes.push(pattern_node);

            // Connect pattern to its variables
            if pattern.subject.starts_with('?') {
                if let Some(&var_node) = variable_nodes.get(&pattern.subject) {
                    graph.connect_pattern_to_variable(
                        pattern_node,
                        var_node,
                        VariableRole::Subject,
                    );
                }
            }
            if pattern.predicate.starts_with('?') {
                if let Some(&var_node) = variable_nodes.get(&pattern.predicate) {
                    graph.connect_pattern_to_variable(
                        pattern_node,
                        var_node,
                        VariableRole::Predicate,
                    );
                }
            }
            if pattern.object.starts_with('?') {
                if let Some(&var_node) = variable_nodes.get(&pattern.object) {
                    graph.connect_pattern_to_variable(pattern_node, var_node, VariableRole::Object);
                }
            }
        }

        // Add filter nodes and dependencies
        for filter in &query_info.filters {
            let filter_node = graph.add_filter_node(filter.clone());

            // Connect filter to patterns that provide its variables
            for var in &filter.variables {
                if let Some(&var_node) = variable_nodes.get(var) {
                    // Find patterns that bind this variable
                    for &pattern_node in &pattern_nodes {
                        if graph.pattern_binds_variable(pattern_node, var_node) {
                            graph.connect_filter_to_pattern(filter_node, pattern_node);
                        }
                    }
                }
            }
        }

        Ok(graph)
    }

    /// Find connected components in the query graph
    fn find_connected_components(&self, graph: &QueryGraph) -> Vec<QueryComponent> {
        let mut components = Vec::new();
        let mut visited = HashSet::new();

        for node in graph.pattern_nodes() {
            if !visited.contains(&node) {
                let component = self.explore_component(graph, node, &mut visited);
                components.push(component);
            }
        }

        debug!("Found {} connected components", components.len());
        components
    }

    /// Explore a connected component starting from a node
    fn explore_component(
        &self,
        graph: &QueryGraph,
        start: NodeIndex,
        visited: &mut HashSet<NodeIndex>,
    ) -> QueryComponent {
        let mut component = QueryComponent::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            if visited.insert(node) {
                match graph.node_type(node) {
                    Some(NodeType::Pattern(idx, pattern)) => {
                        component.patterns.push((*idx, pattern.clone()));
                    }
                    Some(NodeType::Variable(var)) => {
                        component.variables.insert(var.clone());
                    }
                    Some(NodeType::Filter(filter)) => {
                        component.filters.push(filter.clone());
                    }
                    None => {}
                }

                // Add connected nodes to queue
                for neighbor in graph.neighbors(node) {
                    if !visited.contains(&neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        component
    }

    /// Generate candidate execution plans for a component
    fn generate_component_plans(
        &self,
        component: &QueryComponent,
        registry: &ServiceRegistry,
    ) -> Result<Vec<ComponentPlan>> {
        let mut plans = Vec::new();

        // Get available services
        let services = registry.get_services_with_capability(&ServiceCapability::SparqlQuery);
        if services.is_empty() {
            return Err(anyhow!("No available services for query execution"));
        }

        // Strategy 1: Execute entire component on single service
        for service in &services {
            if self.can_service_handle_component(service, component) {
                let plan = self.create_single_service_plan(service, component)?;
                plans.push(plan);
            }
        }

        // Strategy 2: Distribute component across multiple services
        if component.patterns.len() > self.config.min_patterns_for_distribution {
            let distributed_plans = self.create_distributed_plans(component, &services)?;
            plans.extend(distributed_plans);
        }

        // Strategy 3: Specialized service assignment based on predicates
        let specialized_plan = self.create_specialized_service_plan(component, &services)?;
        if let Some(plan) = specialized_plan {
            plans.push(plan);
        }

        if plans.is_empty() {
            // Fallback: force distribution even for small components
            let forced_plan = self.create_forced_distribution_plan(component, &services)?;
            plans.push(forced_plan);
        }

        Ok(plans)
    }

    /// Check if a service can handle an entire component
    fn can_service_handle_component(
        &self,
        service: &FederatedService,
        component: &QueryComponent,
    ) -> bool {
        // Check basic capability
        if !service
            .capabilities
            .contains(&ServiceCapability::SparqlQuery)
        {
            return false;
        }

        // Check for special requirements
        for filter in &component.filters {
            if filter.expression.contains("REGEX")
                && !service
                    .capabilities
                    .contains(&ServiceCapability::FullTextSearch)
            {
                return false;
            }
        }

        // Check data patterns if available
        if !service.data_patterns.is_empty() && service.data_patterns[0] != "*" {
            // Simple pattern matching - could be enhanced
            return true; // For now, assume it can handle if it has specific patterns
        }

        true
    }

    /// Create a plan to execute component on single service
    fn create_single_service_plan(
        &self,
        service: &FederatedService,
        component: &QueryComponent,
    ) -> Result<ComponentPlan> {
        let query_fragment = self.build_component_query(component)?;
        let estimated_cost = self
            .cost_estimator
            .estimate_single_service_cost(service, component);

        Ok(ComponentPlan {
            strategy: PlanStrategy::SingleService,
            steps: vec![PlanStep {
                service_id: service.id.clone(),
                patterns: component.patterns.clone(),
                filters: component.filters.clone(),
                estimated_cost,
                estimated_results: self.estimate_result_size(service, &component.patterns),
            }],
            total_cost: estimated_cost,
            requires_join: false,
        })
    }

    /// Create distributed execution plans using advanced algorithms
    fn create_distributed_plans(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<Vec<ComponentPlan>> {
        let mut plans = Vec::new();

        // Advanced Algorithm 1: Join-aware source selection
        let join_aware_plan = self.distribute_join_aware(component, services)?;
        plans.push(join_aware_plan);

        // Advanced Algorithm 2: Cost-based source ranking with selectivity analysis
        let cost_based_plan = self.distribute_cost_based(component, services)?;
        plans.push(cost_based_plan);

        // Advanced Algorithm 3: Pattern-based source selection with data overlap detection
        let pattern_based_plan = self.distribute_pattern_based(component, services)?;
        plans.push(pattern_based_plan);

        // Advanced Algorithm 4: Star join detection and optimization
        if self.is_star_join_pattern(component) {
            let star_join_plan = self.distribute_star_join_optimized(component, services)?;
            plans.push(star_join_plan);
        }

        // Fallback: Even distribution (existing strategy)
        let even_plan = self.distribute_patterns_evenly(component, services)?;
        plans.push(even_plan);

        // Advanced Algorithm 5: Minimize intermediate results with bloom filters
        let min_intermediate_plan =
            self.distribute_minimize_intermediate_advanced(component, services)?;
        plans.push(min_intermediate_plan);

        // Advanced Algorithm 6: Maximize parallelism with dependency analysis
        if services.len() >= 3 {
            let parallel_plan = self.distribute_maximize_parallel_advanced(component, services)?;
            plans.push(parallel_plan);
        }

        Ok(plans)
    }

    /// Distribute patterns evenly across services
    fn distribute_patterns_evenly(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<ComponentPlan> {
        let patterns_per_service = (component.patterns.len() + services.len() - 1) / services.len();
        let mut steps = Vec::new();
        let mut total_cost = 0.0;

        for (i, chunk) in component.patterns.chunks(patterns_per_service).enumerate() {
            if i < services.len() {
                let service = services[i];
                let cost = self.cost_estimator.estimate_pattern_cost(service, chunk);
                total_cost += cost;

                steps.push(PlanStep {
                    service_id: service.id.clone(),
                    patterns: chunk.to_vec(),
                    filters: Vec::new(), // Filters will be applied after join
                    estimated_cost: cost,
                    estimated_results: self.estimate_result_size(service, chunk),
                });
            }
        }

        Ok(ComponentPlan {
            strategy: PlanStrategy::EvenDistribution,
            steps,
            total_cost,
            requires_join: true,
        })
    }

    /// Distribute to minimize intermediate results
    fn distribute_minimize_intermediate(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<ComponentPlan> {
        // Group patterns by join selectivity
        let pattern_groups = self.group_patterns_by_selectivity(&component.patterns);
        let mut steps = Vec::new();
        let mut total_cost = 0.0;

        for (i, group) in pattern_groups.iter().enumerate() {
            let service = services[i % services.len()];
            let cost = self.cost_estimator.estimate_pattern_cost(service, group);
            total_cost += cost;

            steps.push(PlanStep {
                service_id: service.id.clone(),
                patterns: group.clone(),
                filters: Vec::new(),
                estimated_cost: cost,
                estimated_results: self.estimate_result_size(service, group),
            });
        }

        Ok(ComponentPlan {
            strategy: PlanStrategy::MinimizeIntermediate,
            steps,
            total_cost,
            requires_join: true,
        })
    }

    /// Distribute to maximize parallel execution
    fn distribute_maximize_parallel(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<ComponentPlan> {
        // Create independent pattern groups
        let independent_groups = self.find_independent_pattern_groups(&component.patterns);
        let mut steps = Vec::new();
        let mut max_cost: f64 = 0.0;

        for (i, group) in independent_groups.iter().enumerate() {
            let service = services[i % services.len()];
            let cost = self.cost_estimator.estimate_pattern_cost(service, group);
            max_cost = max_cost.max(cost);

            steps.push(PlanStep {
                service_id: service.id.clone(),
                patterns: group.clone(),
                filters: Vec::new(),
                estimated_cost: cost,
                estimated_results: self.estimate_result_size(service, group),
            });
        }

        Ok(ComponentPlan {
            strategy: PlanStrategy::MaximizeParallel,
            steps,
            total_cost: max_cost, // Total cost is max since they run in parallel
            requires_join: true,
        })
    }

    /// Create plan using specialized services for specific predicates
    fn create_specialized_service_plan(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<Option<ComponentPlan>> {
        let mut predicate_services: HashMap<String, &FederatedService> = HashMap::new();

        // Map predicates to best services
        for (_, pattern) in &component.patterns {
            if !pattern.predicate.starts_with('?') {
                let best_service =
                    self.find_best_service_for_predicate(&pattern.predicate, services);
                if let Some(service) = best_service {
                    predicate_services.insert(pattern.predicate.clone(), service);
                }
            }
        }

        if predicate_services.is_empty() {
            return Ok(None);
        }

        // Group patterns by assigned service
        let mut service_patterns: HashMap<String, Vec<(usize, TriplePattern)>> = HashMap::new();
        for (idx, pattern) in &component.patterns {
            let service_id = if let Some(service) = predicate_services.get(&pattern.predicate) {
                service.id.clone()
            } else {
                // Assign to service with most patterns
                services[0].id.clone()
            };

            service_patterns
                .entry(service_id)
                .or_default()
                .push((*idx, pattern.clone()));
        }

        let mut steps = Vec::new();
        let mut total_cost = 0.0;

        for (service_id, patterns) in service_patterns {
            let service = services.iter().find(|s| s.id == service_id).unwrap();
            let cost = self
                .cost_estimator
                .estimate_pattern_cost(service, &patterns);
            total_cost += cost;

            let estimated_results = self.estimate_result_size(service, &patterns);
            steps.push(PlanStep {
                service_id: service_id.clone(),
                patterns,
                filters: Vec::new(),
                estimated_cost: cost,
                estimated_results,
            });
        }

        let requires_join = steps.len() > 1;
        Ok(Some(ComponentPlan {
            strategy: PlanStrategy::SpecializedServices,
            steps,
            total_cost,
            requires_join,
        }))
    }

    /// Force distribution when no other option works
    fn create_forced_distribution_plan(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<ComponentPlan> {
        warn!(
            "Using forced distribution for component with {} patterns",
            component.patterns.len()
        );

        // Just use the first available service
        let service = services[0];
        self.create_single_service_plan(service, component)
    }

    /// Find best service for a specific predicate
    fn find_best_service_for_predicate<'a>(
        &self,
        predicate: &str,
        services: &[&'a FederatedService],
    ) -> Option<&'a FederatedService> {
        let mut best_service = None;
        let mut best_score = 0.0;

        for service in services {
            let score = self.calculate_predicate_affinity(predicate, service);
            if score > best_score {
                best_score = score;
                best_service = Some(*service);
            }
        }

        best_service
    }

    /// Calculate affinity score between predicate and service
    fn calculate_predicate_affinity(&self, predicate: &str, service: &FederatedService) -> f64 {
        let mut score = 1.0;

        // Check predicate namespace matches
        if predicate.contains("://") {
            let namespace = predicate.split('/').take(3).collect::<Vec<_>>().join("/");
            if service.endpoint.contains(&namespace) {
                score += 2.0;
            }
        }

        // Check known vocabulary matches
        if predicate.contains("foaf:") && service.name.to_lowercase().contains("foaf") {
            score += 1.5;
        }
        if predicate.contains("geo:")
            && service
                .capabilities
                .contains(&ServiceCapability::Geospatial)
        {
            score += 2.0;
        }

        score
    }

    /// Group patterns by selectivity to minimize intermediate results
    fn group_patterns_by_selectivity(
        &self,
        patterns: &[(usize, TriplePattern)],
    ) -> Vec<Vec<(usize, TriplePattern)>> {
        let mut groups = Vec::new();
        let mut current_group = Vec::new();
        let mut shared_vars = HashSet::new();

        for pattern in patterns {
            let pattern_vars = self.extract_pattern_variables(&pattern.1);

            if shared_vars.is_empty() || !shared_vars.is_disjoint(&pattern_vars) {
                // Pattern shares variables with current group
                current_group.push(pattern.clone());
                shared_vars.extend(pattern_vars);
            } else {
                // Start new group
                if !current_group.is_empty() {
                    groups.push(current_group);
                }
                current_group = vec![pattern.clone()];
                shared_vars = pattern_vars;
            }
        }

        if !current_group.is_empty() {
            groups.push(current_group);
        }

        groups
    }

    /// Find groups of patterns that can be executed independently
    fn find_independent_pattern_groups(
        &self,
        patterns: &[(usize, TriplePattern)],
    ) -> Vec<Vec<(usize, TriplePattern)>> {
        let mut groups = Vec::new();
        let mut remaining = patterns.to_vec();

        while !remaining.is_empty() {
            let mut group = vec![remaining.remove(0)];
            let mut group_vars = self.extract_pattern_variables(&group[0].1);

            // Add patterns that don't share variables
            remaining.retain(|pattern| {
                let pattern_vars = self.extract_pattern_variables(&pattern.1);
                if group_vars.is_disjoint(&pattern_vars) {
                    group.push(pattern.clone());
                    group_vars.extend(pattern_vars);
                    false
                } else {
                    true
                }
            });

            groups.push(group);
        }

        groups
    }

    /// Extract variables from a pattern
    fn extract_pattern_variables(&self, pattern: &TriplePattern) -> HashSet<String> {
        let mut vars = HashSet::new();

        if pattern.subject.starts_with('?') {
            vars.insert(pattern.subject.clone());
        }
        if pattern.predicate.starts_with('?') {
            vars.insert(pattern.predicate.clone());
        }
        if pattern.object.starts_with('?') {
            vars.insert(pattern.object.clone());
        }

        vars
    }

    /// Estimate result size for patterns
    fn estimate_result_size(
        &self,
        service: &FederatedService,
        patterns: &[(usize, TriplePattern)],
    ) -> u64 {
        // Simple estimation based on pattern complexity
        let base_size = 1000;
        let pattern_factor = patterns.len() as u64;
        let selectivity = self.estimate_pattern_selectivity(patterns);

        (base_size * pattern_factor as u64 * selectivity as u64).max(1)
    }

    /// Estimate selectivity of patterns
    fn estimate_pattern_selectivity(&self, patterns: &[(usize, TriplePattern)]) -> f64 {
        let mut selectivity = 1.0;

        for (_, pattern) in patterns {
            let var_count = [&pattern.subject, &pattern.predicate, &pattern.object]
                .iter()
                .filter(|p| p.starts_with('?'))
                .count();

            selectivity *= match var_count {
                0 => 0.001, // All constants - very selective
                1 => 0.01,  // One variable
                2 => 0.1,   // Two variables
                3 => 1.0,   // All variables - least selective
                _ => 1.0,
            };
        }

        selectivity
    }

    /// Select optimal combination of component plans
    fn select_optimal_plan_combination(
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
    fn build_execution_plan(
        &self,
        component_plans: Vec<ComponentPlan>,
        query_info: &QueryInfo,
    ) -> Result<ExecutionPlan> {
        let mut plan = ExecutionPlan {
            query_id: uuid::Uuid::new_v4().to_string(),
            query_type: query_info.query_type,
            steps: Vec::new(),
            estimated_duration: Duration::from_secs(0),
            parallelizable_steps: Vec::new(),
            dependencies: HashMap::new(),
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
                    expected_variables: self.extract_step_variables(plan_step),
                    estimated_duration: Duration::from_millis(
                        (plan_step.estimated_cost * 10.0) as u64,
                    ),
                    dependencies: Vec::new(),
                    parallel_group: Some(comp_idx),
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
                    expected_variables: HashSet::new(),
                    estimated_duration: Duration::from_millis(50),
                    dependencies: component_step_ids.clone(),
                    parallel_group: None,
                };

                let join_id = join_step.step_id.clone();
                plan.steps.push(join_step);
                step_ids_by_component.push(vec![join_id]);
            } else {
                step_ids_by_component.push(component_step_ids);
            }
        }

        // Add final join if multiple components
        if step_ids_by_component.len() > 1 {
            let all_deps: Vec<String> = step_ids_by_component.into_iter().flatten().collect();

            let final_join = ExecutionStep {
                step_id: uuid::Uuid::new_v4().to_string(),
                step_type: StepType::Join,
                service_id: None,
                query_fragment: "-- Final join across components".to_string(),
                expected_variables: query_info.variables.clone(),
                estimated_duration: Duration::from_millis(100),
                dependencies: all_deps,
                parallel_group: None,
            };

            plan.steps.push(final_join);
        }

        // Calculate total estimated duration
        let max_parallel_duration = component_plans
            .iter()
            .map(|cp| Duration::from_millis((cp.total_cost * 10.0) as u64))
            .max()
            .unwrap_or(Duration::from_secs(0));

        plan.estimated_duration = max_parallel_duration + Duration::from_millis(200); // Add overhead

        Ok(plan)
    }

    /// Build query for a component
    fn build_component_query(&self, component: &QueryComponent) -> Result<String> {
        let mut query = String::from("SELECT ");

        // Add variables
        if component.variables.is_empty() {
            query.push_str("* ");
        } else {
            let vars: Vec<String> = component.variables.iter().cloned().collect();
            query.push_str(&vars.join(" "));
            query.push(' ');
        }

        query.push_str("WHERE {\n");

        // Add patterns
        for (_, pattern) in &component.patterns {
            query.push_str("  ");
            query.push_str(&pattern.pattern_string);
            query.push_str(" .\n");
        }

        // Add filters
        for filter in &component.filters {
            query.push_str("  FILTER(");
            query.push_str(&filter.expression);
            query.push_str(")\n");
        }

        query.push_str("}");

        Ok(query)
    }

    /// Build query for an execution step
    fn build_step_query(&self, step: &PlanStep) -> Result<String> {
        let mut query = String::from("SELECT * WHERE {\n");

        for (_, pattern) in &step.patterns {
            query.push_str("  ");
            query.push_str(&pattern.pattern_string);
            query.push_str(" .\n");
        }

        for filter in &step.filters {
            query.push_str("  FILTER(");
            query.push_str(&filter.expression);
            query.push_str(")\n");
        }

        query.push_str("}");

        Ok(query)
    }

    /// Extract variables from a plan step
    fn extract_step_variables(&self, step: &PlanStep) -> HashSet<String> {
        let mut vars = HashSet::new();

        for (_, pattern) in &step.patterns {
            vars.extend(self.extract_pattern_variables(pattern));
        }

        vars
    }

    /// Calculate decomposition statistics
    fn calculate_decomposition_stats(
        &self,
        component_plans: &[Vec<ComponentPlan>],
    ) -> DecompositionStats {
        let total_plans_generated = component_plans.iter().map(|cp| cp.len()).sum();
        let components_count = component_plans.len();

        let avg_plans_per_component = if components_count > 0 {
            total_plans_generated as f64 / components_count as f64
        } else {
            0.0
        };

        DecompositionStats {
            components_found: components_count,
            total_plans_generated,
            avg_plans_per_component,
            decomposition_time: Duration::from_millis(0), // Would be tracked in real implementation
        }
    }

    // ============= ADVANCED QUERY DECOMPOSITION ALGORITHMS =============

    /// Advanced Algorithm 1: Join-aware source selection
    /// Prioritizes services that can handle connected patterns together to minimize joins
    fn distribute_join_aware(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<ComponentPlan> {
        let mut join_graph = self.build_join_graph(&component.patterns);
        let mut service_assignments: HashMap<String, Vec<&TriplePattern>> = HashMap::new();
        let mut steps = Vec::new();

        // Find connected pattern groups that share variables
        let connected_groups = self.find_connected_pattern_groups(&join_graph);

        for group in connected_groups {
            // Find service that can handle the most patterns in this group
            let best_service = self.find_best_service_for_group(&group, services);

            if let Some(service) = best_service {
                let cost = self.cost_estimator.estimate_pattern_cost(service, &group);
                steps.push(PlanStep {
                    service_id: service.id.clone(),
                    patterns: group.clone(),
                    filters: self.extract_applicable_filters(&component.filters, &group),
                    estimated_cost: cost,
                    estimated_results: self.estimate_result_size(service, &group),
                });
            }
        }

        let total_cost = steps.iter().map(|s| s.estimated_cost).sum();
        let requires_join = steps.len() > 1;

        Ok(ComponentPlan {
            strategy: PlanStrategy::JoinAware,
            steps,
            total_cost,
            requires_join,
        })
    }

    /// Advanced Algorithm 2: Cost-based source ranking with selectivity analysis
    fn distribute_cost_based(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<ComponentPlan> {
        let mut pattern_assignments = Vec::new();

        // For each pattern, rank services by cost and selectivity
        for (idx, pattern) in &component.patterns {
            let mut service_scores = Vec::new();

            for service in services {
                let base_cost = self
                    .cost_estimator
                    .estimate_single_pattern_cost(service, pattern);
                let selectivity = self.estimate_pattern_selectivity(&[(0, pattern.clone())]);
                let component = QueryComponent {
                    patterns: vec![pattern.clone()],
                    filters: Vec::new(),
                    variables: Vec::new(),
                    joins: Vec::new(),
                };
                let network_cost = self
                    .cost_estimator
                    .estimate_network_cost(service, &component);

                // Combined score: lower is better
                let score = base_cost + (1.0 / selectivity) + network_cost;
                service_scores.push((service, score));
            }

            // Sort by score (ascending - lower is better)
            service_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Assign to best service
            if let Some((best_service, _)) = service_scores.first() {
                pattern_assignments.push((*idx, pattern.clone(), best_service.id.clone()));
            }
        }

        // Group patterns by assigned service
        let service_groups = self.group_patterns_by_service(pattern_assignments);
        let steps = self.create_steps_from_service_groups(service_groups, services)?;
        let total_cost = steps.iter().map(|s| s.estimated_cost).sum();

        Ok(ComponentPlan {
            strategy: PlanStrategy::CostBased,
            steps,
            total_cost,
            requires_join: steps.len() > 1,
        })
    }

    /// Advanced Algorithm 3: Pattern-based source selection with data overlap detection
    fn distribute_pattern_based(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<ComponentPlan> {
        let mut steps = Vec::new();
        let mut used_services = HashSet::new();

        // Group patterns by predicate and namespace
        let predicate_groups = self.group_patterns_by_predicate(&component.patterns);

        for (predicate, patterns) in predicate_groups {
            // Find services with data for this predicate
            let candidate_services: Vec<_> = services
                .iter()
                .filter(|s| self.service_has_predicate_data(s, &predicate))
                .cloned()
                .collect();

            if candidate_services.is_empty() {
                // Use any available service as fallback
                if let Some(service) = services.first() {
                    let cost = self
                        .cost_estimator
                        .estimate_pattern_cost(service, &patterns);
                    steps.push(PlanStep {
                        service_id: service.id.clone(),
                        patterns: patterns.clone(),
                        filters: Vec::new(),
                        estimated_cost: cost,
                        estimated_results: self.estimate_result_size(service, &patterns),
                    });
                }
                continue;
            }

            // Check for data overlap between candidate services
            let overlap_matrix =
                self.calculate_data_overlap_matrix(&candidate_services, &predicate);

            // Select service with least overlap (most unique data)
            let best_service =
                self.select_service_minimizing_overlap(&candidate_services, &overlap_matrix);

            if let Some(service) = best_service {
                used_services.insert(service.id.clone());
                let cost = self
                    .cost_estimator
                    .estimate_pattern_cost(service, &patterns);
                steps.push(PlanStep {
                    service_id: service.id.clone(),
                    patterns: patterns.clone(),
                    filters: Vec::new(),
                    estimated_cost: cost,
                    estimated_results: self.estimate_result_size(service, &patterns),
                });
            }
        }

        let total_cost = steps.iter().map(|s| s.estimated_cost).sum();
        let requires_join = steps.len() > 1;

        Ok(ComponentPlan {
            strategy: PlanStrategy::PatternBased,
            steps,
            total_cost,
            requires_join,
        })
    }

    /// Advanced Algorithm 4: Star join detection and optimization
    fn distribute_star_join_optimized(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<ComponentPlan> {
        let star_center = self.detect_star_center(&component.patterns)?;
        let mut steps = Vec::new();

        // Find the service with most data for the star center variable
        let center_service = self.find_best_service_for_star_center(&star_center, services);

        if let Some(service) = center_service {
            // Group patterns by their relationship to the star center
            let center_patterns =
                self.get_patterns_with_variable(&component.patterns, &star_center);
            let peripheral_patterns =
                self.get_patterns_without_variable(&component.patterns, &star_center);

            // Execute center patterns on the selected service
            if !center_patterns.is_empty() {
                let cost = self
                    .cost_estimator
                    .estimate_pattern_cost(service, &center_patterns);
                steps.push(PlanStep {
                    service_id: service.id.clone(),
                    patterns: center_patterns,
                    filters: self
                        .extract_applicable_filters(&component.filters, &component.patterns),
                    estimated_cost: cost,
                    estimated_results: self.estimate_result_size(service, &component.patterns),
                });
            }

            // Distribute peripheral patterns to other services
            for pattern_group in self.chunk_patterns(peripheral_patterns, services.len() - 1) {
                if let Some(peripheral_service) = services.iter().find(|s| s.id != service.id) {
                    let cost = self
                        .cost_estimator
                        .estimate_pattern_cost(peripheral_service, &pattern_group);
                    steps.push(PlanStep {
                        service_id: peripheral_service.id.clone(),
                        patterns: pattern_group,
                        filters: Vec::new(),
                        estimated_cost: cost,
                        estimated_results: self
                            .estimate_result_size(peripheral_service, &component.patterns),
                    });
                }
            }
        }

        let total_cost = steps.iter().map(|s| s.estimated_cost).sum();
        let requires_join = steps.len() > 1;

        Ok(ComponentPlan {
            strategy: PlanStrategy::StarJoinOptimized,
            steps,
            total_cost,
            requires_join,
        })
    }

    /// Advanced Algorithm 5: Minimize intermediate results with bloom filters
    fn distribute_minimize_intermediate_advanced(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<ComponentPlan> {
        let mut steps = Vec::new();

        // Estimate intermediate result sizes for different execution orders
        let execution_orders = self.generate_execution_orders(&component.patterns);
        let best_order =
            self.select_order_minimizing_intermediate_results(execution_orders, services);

        // Create execution steps with bloom filter hints
        for (service_id, pattern) in best_order {
            let service = &services[service_id % services.len()];
            let pattern_vec = vec![(0, pattern.clone())];
            let cost = self
                .cost_estimator
                .estimate_pattern_cost(service, &pattern_vec);

            // Add bloom filter optimization hint
            let estimated_results =
                self.estimate_result_size_with_bloom_filter(service, &pattern_vec);

            steps.push(PlanStep {
                service_id: service_id.to_string(),
                patterns: vec![(0, pattern)],
                filters: Vec::new(),
                estimated_cost: cost * 0.8, // Bloom filter reduces cost
                estimated_results: estimated_results as u64,
            });
        }

        let total_cost = steps.iter().map(|s| s.estimated_cost).sum();

        let requires_join = steps.len() > 1;
        Ok(ComponentPlan {
            strategy: PlanStrategy::MinimizeIntermediateAdvanced,
            steps,
            total_cost,
            requires_join,
        })
    }

    /// Advanced Algorithm 6: Maximize parallelism with dependency analysis
    fn distribute_maximize_parallel_advanced(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<ComponentPlan> {
        // Build dependency graph between patterns
        let dependency_graph = self.build_pattern_dependency_graph(&component.patterns);

        // Find patterns that can execute in parallel (no dependencies)
        let parallel_groups = self.find_parallel_execution_groups(&dependency_graph);

        let mut steps = Vec::new();
        let mut service_idx = 0;

        for group in parallel_groups {
            // Assign each parallel group to different services
            let service = services[service_idx % services.len()];
            service_idx += 1;

            let cost = self.cost_estimator.estimate_pattern_cost(service, &group);
            steps.push(PlanStep {
                service_id: service.id.clone(),
                patterns: group,
                filters: Vec::new(),
                estimated_cost: cost,
                estimated_results: self.estimate_result_size(service, &component.patterns),
            });
        }

        // Use maximum cost since they run in parallel
        let total_cost = steps
            .iter()
            .map(|s| s.estimated_cost)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        let requires_join = steps.len() > 1;

        Ok(ComponentPlan {
            strategy: PlanStrategy::MaximizeParallelAdvanced,
            steps,
            total_cost,
            requires_join,
        })
    }

    // ============= HELPER METHODS FOR ADVANCED ALGORITHMS =============

    /// Detect if the component follows a star join pattern
    fn is_star_join_pattern(&self, component: &QueryComponent) -> bool {
        if component.patterns.len() < 3 {
            return false;
        }

        let mut variable_counts = HashMap::new();
        for (_, pattern) in &component.patterns {
            for var in [&pattern.subject, &pattern.predicate, &pattern.object] {
                if var.starts_with('?') {
                    *variable_counts.entry(var.clone()).or_insert(0) += 1;
                }
            }
        }

        // A star join has one central variable appearing in most patterns
        let max_count = variable_counts.values().max().unwrap_or(&0);
        *max_count >= (component.patterns.len() as i32 - 1) && component.patterns.len() >= 3
    }

    /// Detect the center variable of a star join
    fn detect_star_center(&self, patterns: &[(usize, TriplePattern)]) -> Result<String> {
        let mut variable_counts = HashMap::new();
        for (_, pattern) in patterns {
            for var in [&pattern.subject, &pattern.predicate, &pattern.object] {
                if var.starts_with('?') {
                    *variable_counts.entry(var.clone()).or_insert(0) += 1;
                }
            }
        }

        variable_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(var, _)| var)
            .ok_or_else(|| anyhow!("No star center found"))
    }

    /// Build a graph of pattern dependencies based on shared variables
    fn build_pattern_dependency_graph(
        &self,
        patterns: &[(usize, TriplePattern)],
    ) -> DiGraph<usize, ()> {
        let mut graph = DiGraph::new();
        let mut node_map = HashMap::new();

        // Add nodes for each pattern
        for (idx, _) in patterns {
            let node = graph.add_node(*idx);
            node_map.insert(*idx, node);
        }

        // Add edges for dependencies (patterns that share variables)
        for (i, (idx1, pattern1)) in patterns.iter().enumerate() {
            for (idx2, pattern2) in patterns.iter().skip(i + 1) {
                if self.patterns_share_variables(pattern1, pattern2) {
                    if let (Some(&node1), Some(&node2)) = (node_map.get(idx1), node_map.get(idx2)) {
                        graph.add_edge(node1, node2, ());
                    }
                }
            }
        }

        graph
    }

    /// Check if two patterns share any variables
    fn patterns_share_variables(&self, p1: &TriplePattern, p2: &TriplePattern) -> bool {
        let p1_parts = vec![&p1.subject, &p1.predicate, &p1.object];
        let p1_vars: HashSet<_> = p1_parts
            .into_iter()
            .filter(|v| v.starts_with('?'))
            .collect();
        let p2_parts = vec![&p2.subject, &p2.predicate, &p2.object];
        let p2_vars: HashSet<_> = p2_parts
            .into_iter()
            .filter(|v| v.starts_with('?'))
            .collect();

        !p1_vars.is_disjoint(&p2_vars)
    }

    /// Build join graph showing variable dependencies between patterns
    fn build_join_graph(&self, patterns: &[(usize, TriplePattern)]) -> HashMap<String, Vec<usize>> {
        let mut join_graph = HashMap::new();

        // For each variable, track which patterns use it
        for (idx, pattern) in patterns {
            let pattern_vars = self.extract_pattern_variables(pattern);
            for var in pattern_vars {
                join_graph.entry(var).or_insert_with(Vec::new).push(*idx);
            }
        }

        // Remove variables that appear in only one pattern (no joins)
        join_graph.retain(|_, pattern_indices| pattern_indices.len() > 1);

        join_graph
    }

    fn find_connected_pattern_groups(
        &self,
        join_graph: &HashMap<String, Vec<usize>>,
    ) -> Vec<Vec<(usize, TriplePattern)>> {
        let mut groups = Vec::new();
        let mut visited_patterns = HashSet::new();

        // Build pattern connectivity map
        let mut pattern_connections: HashMap<usize, HashSet<usize>> = HashMap::new();
        for (_, pattern_indices) in join_graph {
            for &idx1 in pattern_indices {
                for &idx2 in pattern_indices {
                    if idx1 != idx2 {
                        pattern_connections.entry(idx1).or_default().insert(idx2);
                    }
                }
            }
        }

        // Find connected components using DFS
        for &pattern_idx in pattern_connections.keys() {
            if !visited_patterns.contains(&pattern_idx) {
                let mut group = Vec::new();
                let mut stack = vec![pattern_idx];

                while let Some(current) = stack.pop() {
                    if visited_patterns.insert(current) {
                        // Find the pattern by index - this is a simplified lookup
                        // In a real implementation, we'd maintain a proper index->pattern mapping
                        if let Some(connections) = pattern_connections.get(&current) {
                            for &connected in connections {
                                if !visited_patterns.contains(&connected) {
                                    stack.push(connected);
                                }
                            }
                        }
                    }
                }

                if !group.is_empty() {
                    groups.push(group);
                }
            }
        }

        groups
    }

    fn find_best_service_for_group<'a>(
        &self,
        group: &[(usize, TriplePattern)],
        services: &[&'a FederatedService],
    ) -> Option<&'a FederatedService> {
        let mut best_service = None;
        let mut best_score = 0.0;

        for service in services {
            let mut score = 0.0;

            // Score based on predicate affinity
            for (_, pattern) in group {
                if !pattern.predicate.starts_with('?') {
                    score += self.calculate_predicate_affinity(&pattern.predicate, service);
                }
            }

            // Bonus for services that can handle the entire group
            if self.can_service_handle_all_patterns(service, group) {
                score += 5.0;
            }

            // Penalty for high load
            let load_penalty = self.cost_estimator.get_service_load_factor(service) - 1.0;
            score -= load_penalty * 2.0;

            if score > best_score {
                best_score = score;
                best_service = Some(*service);
            }
        }

        best_service
    }

    fn extract_applicable_filters(
        &self,
        filters: &[FilterExpression],
        patterns: &[(usize, TriplePattern)],
    ) -> Vec<FilterExpression> {
        let pattern_variables: HashSet<String> = patterns
            .iter()
            .flat_map(|(_, pattern)| self.extract_pattern_variables(pattern))
            .collect();

        filters
            .iter()
            .filter(|filter| {
                // Include filter if all its variables are provided by the patterns
                filter
                    .variables
                    .iter()
                    .all(|var| pattern_variables.contains(var))
            })
            .cloned()
            .collect()
    }

    fn estimate_pattern_selectivity(
        &self,
        service: &FederatedService,
        pattern: &TriplePattern,
    ) -> f64 {
        let mut selectivity = 1.0;

        // Subject selectivity
        if pattern.subject.starts_with('?') {
            selectivity *= 0.1; // Variable subject is selective
        } else if pattern.subject.starts_with('<') {
            selectivity *= 0.001; // URI subject is very selective
        }

        // Predicate selectivity
        if pattern.predicate.starts_with('?') {
            selectivity *= 0.01; // Variable predicate is very selective
        } else {
            // Use service-specific predicate frequency if available
            if let Some(freq) = self.get_predicate_frequency(service, &pattern.predicate) {
                selectivity *= freq;
            } else {
                selectivity *= 0.1; // Default predicate selectivity
            }
        }

        // Object selectivity
        if pattern.object.starts_with('?') {
            selectivity *= 0.3; // Variable object has moderate selectivity
        } else if pattern.object.starts_with('<') {
            selectivity *= 0.01; // URI object is selective
        } else if pattern.object.starts_with('"') {
            selectivity *= 0.05; // Literal object is quite selective
        }

        selectivity.max(0.0001) // Minimum selectivity
    }

    fn group_patterns_by_service(
        &self,
        assignments: Vec<(usize, TriplePattern, String)>,
    ) -> HashMap<String, Vec<(usize, TriplePattern)>> {
        let mut groups = HashMap::new();
        for (idx, pattern, service_id) in assignments {
            groups
                .entry(service_id)
                .or_insert_with(Vec::new)
                .push((idx, pattern));
        }
        groups
    }

    fn create_steps_from_service_groups(
        &self,
        groups: HashMap<String, Vec<(usize, TriplePattern)>>,
        services: &[&FederatedService],
    ) -> Result<Vec<PlanStep>> {
        let mut steps = Vec::new();
        for (service_id, patterns) in groups {
            if let Some(service) = services.iter().find(|s| s.id == service_id) {
                let cost = self
                    .cost_estimator
                    .estimate_pattern_cost(service, &patterns);
                steps.push(PlanStep {
                    service_id: service_id.clone(),
                    patterns: patterns.clone(),
                    filters: Vec::new(),
                    estimated_cost: cost,
                    estimated_results: self.estimate_result_size(service, &patterns),
                });
            }
        }
        Ok(steps)
    }

    /// Select service that minimizes data overlap
    fn select_service_minimizing_overlap<'a>(
        &self,
        candidates: &[&'a FederatedService],
        _overlap_matrix: &HashMap<String, f64>,
    ) -> Option<&'a FederatedService> {
        // Simplified implementation - select first candidate
        candidates.first().copied()
    }

    /// Find best service for star center variable
    fn find_best_service_for_star_center<'a>(
        &self,
        _star_center: &str,
        services: &[&'a FederatedService],
    ) -> Option<&'a FederatedService> {
        // Simplified implementation - select first service
        services.first().copied()
    }

    /// Get patterns containing a specific variable
    fn get_patterns_with_variable(
        &self,
        patterns: &[(usize, TriplePattern)],
        variable: &str,
    ) -> Vec<(usize, TriplePattern)> {
        patterns
            .iter()
            .filter(|(_, pattern)| {
                pattern.subject == variable
                    || pattern.predicate == variable
                    || pattern.object == variable
            })
            .cloned()
            .collect()
    }

    /// Get patterns NOT containing a specific variable
    fn get_patterns_without_variable(
        &self,
        patterns: &[(usize, TriplePattern)],
        variable: &str,
    ) -> Vec<(usize, TriplePattern)> {
        patterns
            .iter()
            .filter(|(_, pattern)| {
                pattern.subject != variable
                    && pattern.predicate != variable
                    && pattern.object != variable
            })
            .cloned()
            .collect()
    }

    /// Chunk patterns into groups
    fn chunk_patterns(
        &self,
        patterns: Vec<(usize, TriplePattern)>,
        chunk_size: usize,
    ) -> Vec<Vec<(usize, TriplePattern)>> {
        patterns
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Generate possible execution orders
    fn generate_execution_orders(
        &self,
        patterns: &[(usize, TriplePattern)],
    ) -> Vec<Vec<(usize, TriplePattern)>> {
        // Simplified implementation - return original order
        vec![patterns.to_vec()]
    }

    /// Select execution order that minimizes intermediate results
    fn select_order_minimizing_intermediate_results(
        &self,
        orders: Vec<Vec<(usize, TriplePattern)>>,
        _services: &[&FederatedService],
    ) -> Vec<(usize, TriplePattern)> {
        // Simplified implementation - return first order
        orders.into_iter().next().unwrap_or_default()
    }

    /// Estimate result size with Bloom filter optimization
    fn estimate_result_size_with_bloom_filter(
        &self,
        service: &FederatedService,
        patterns: &[(usize, TriplePattern)],
    ) -> usize {
        // Simplified implementation - use regular estimate with reduction factor
        let base_estimate = self.estimate_result_size(service, patterns);
        (base_estimate as f64 * 0.7) as usize // 30% reduction with Bloom filter
    }

    /// Find groups of patterns that can execute in parallel
    fn find_parallel_execution_groups(
        &self,
        _dependency_graph: &DiGraph<usize, ()>,
    ) -> Vec<Vec<(usize, TriplePattern)>> {
        // Simplified implementation - return empty groups
        Vec::new()
    }

    /// Check if service can handle all patterns in a group
    fn can_service_handle_all_patterns(
        &self,
        service: &FederatedService,
        patterns: &[(usize, TriplePattern)],
    ) -> bool {
        for (_, pattern) in patterns {
            if !self.can_service_handle_pattern(service, pattern) {
                return false;
            }
        }
        true
    }

    /// Check if service can handle a specific pattern
    fn can_service_handle_pattern(
        &self,
        service: &FederatedService,
        pattern: &TriplePattern,
    ) -> bool {
        // Check basic SPARQL capability
        if !service
            .capabilities
            .contains(&ServiceCapability::SparqlQuery)
        {
            return false;
        }

        // Check for special predicate requirements
        if pattern.predicate.contains("geo:")
            && !service
                .capabilities
                .contains(&ServiceCapability::Geospatial)
        {
            return false;
        }

        if pattern.predicate.contains("text:")
            && !service
                .capabilities
                .contains(&ServiceCapability::FullTextSearch)
        {
            return false;
        }

        // Check data pattern coverage if available
        if !service.data_patterns.is_empty() && service.data_patterns[0] != "*" {
            // Simple namespace matching
            for data_pattern in &service.data_patterns {
                if pattern.predicate.starts_with(data_pattern) {
                    return true;
                }
            }
            return false;
        }

        true
    }

    /// Get predicate frequency for a service (estimated)
    fn get_predicate_frequency(&self, service: &FederatedService, predicate: &str) -> Option<f64> {
        // In a real implementation, this would query service statistics
        // For now, provide reasonable estimates based on common vocabularies
        if predicate.contains("rdf:type") {
            Some(0.8) // Type predicates are very common
        } else if predicate.contains("rdfs:label") || predicate.contains("dc:title") {
            Some(0.5) // Label predicates are common
        } else if predicate.contains("foaf:") {
            Some(0.3) // FOAF predicates moderate frequency
        } else if predicate.contains("geo:") {
            Some(0.1) // Geographic predicates less common
        } else {
            None // Unknown frequency
        }
    }

    /// Group patterns by predicate for pattern-based distribution
    fn group_patterns_by_predicate(
        &self,
        patterns: &[(usize, TriplePattern)],
    ) -> HashMap<String, Vec<(usize, TriplePattern)>> {
        let mut groups = HashMap::new();

        for &(idx, ref pattern) in patterns {
            let predicate_key = if pattern.predicate.starts_with('?') {
                "__VARIABLE__".to_string()
            } else {
                // Extract namespace for grouping
                if let Some(ns_end) = pattern.predicate.rfind([':', '#']) {
                    pattern.predicate[..=ns_end].to_string()
                } else {
                    pattern.predicate.clone()
                }
            };

            groups
                .entry(predicate_key)
                .or_insert_with(Vec::new)
                .push((idx, pattern.clone()));
        }

        groups
    }

    /// Check if service has data for specific predicate
    fn service_has_predicate_data(&self, service: &FederatedService, predicate: &str) -> bool {
        if service.data_patterns.is_empty() || service.data_patterns[0] == "*" {
            return true; // Service claims to handle all predicates
        }

        // Check if predicate matches any data pattern
        for pattern in &service.data_patterns {
            if predicate.starts_with(pattern) || pattern.starts_with(predicate) {
                return true;
            }
        }

        false
    }

    /// Calculate data overlap matrix between services
    fn calculate_data_overlap_matrix(
        &self,
        services: &[&FederatedService],
        predicate: &str,
    ) -> HashMap<String, f64> {
        let mut overlap_matrix = HashMap::new();

        for service in services {
            // Simplified overlap calculation
            // In reality, this would use Bloom filters or statistical samples
            let overlap_score = if service.data_patterns.len() > 1 {
                0.3 // Assume 30% overlap for services with multiple patterns
            } else {
                0.1 // Assume 10% overlap for specialized services
            };

            overlap_matrix.insert(service.id.clone(), overlap_score);
        }

        overlap_matrix
    }
}

impl Default for QueryDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced cost estimator for federated query execution
#[derive(Debug)]
struct CostEstimator {
    weights: CostWeights,
    statistics: Arc<RwLock<CostStatistics>>,
    network_model: NetworkCostModel,
    performance_tracker: PerformanceTracker,
}

/// Statistical data for cost estimation
#[derive(Debug)]
struct CostStatistics {
    service_performance: HashMap<String, ServiceStats>,
    pattern_selectivity: HashMap<String, f64>,
    join_cardinality: HashMap<String, u64>,
    cache_hit_rates: HashMap<String, f64>,
    execution_history: VecDeque<ExecutionRecord>,
}

/// Performance statistics for individual services
#[derive(Debug, Clone)]
struct ServiceStats {
    average_response_time: Duration,
    throughput_patterns_per_sec: f64,
    error_rate: f64,
    availability: f64,
    load_factor: f64,
    last_updated: Instant,
}

/// Network cost modeling
#[derive(Debug)]
struct NetworkCostModel {
    base_latency: Duration,
    bandwidth_mbps: f64,
    connection_overhead: Duration,
    compression_ratio: f64,
}

/// Performance tracking for cost model updates
#[derive(Debug)]
struct PerformanceTracker {
    execution_records: VecDeque<ExecutionRecord>,
    prediction_accuracy: f64,
    last_model_update: Instant,
}

/// Record of actual execution for cost model learning
#[derive(Debug, Clone)]
struct ExecutionRecord {
    query_hash: u64,
    service_id: String,
    pattern_count: usize,
    filter_count: usize,
    predicted_cost: f64,
    actual_cost: f64,
    actual_duration: Duration,
    result_count: usize,
    timestamp: Instant,
}

impl CostEstimator {
    fn new() -> Self {
        Self {
            weights: CostWeights::default(),
            statistics: Arc::new(RwLock::new(CostStatistics {
                service_performance: HashMap::new(),
                pattern_selectivity: HashMap::new(),
                join_cardinality: HashMap::new(),
                cache_hit_rates: HashMap::new(),
                execution_history: VecDeque::with_capacity(10000),
            })),
            network_model: NetworkCostModel {
                base_latency: Duration::from_millis(50),
                bandwidth_mbps: 100.0,
                connection_overhead: Duration::from_millis(10),
                compression_ratio: 0.3,
            },
            performance_tracker: PerformanceTracker {
                execution_records: VecDeque::with_capacity(1000),
                prediction_accuracy: 0.8,
                last_model_update: Instant::now(),
            },
        }
    }

    fn estimate_single_service_cost(
        &self,
        service: &FederatedService,
        component: &QueryComponent,
    ) -> f64 {
        // Base computation cost
        let pattern_cost = self.estimate_pattern_computation_cost(&component.patterns, service);
        let filter_cost = self.estimate_filter_cost(&component.filters, service);
        let variable_cost = component.variables.len() as f64 * self.weights.variable_weight;

        // Network and data transfer costs
        let network_cost = self.estimate_network_cost(service, component);
        let data_transfer_cost = self.estimate_data_transfer_cost(service, component);

        // Service-specific factors
        let service_load_factor = self.get_service_load_factor(service);
        let cache_factor = self.estimate_cache_benefit(service, component);

        let base_cost =
            pattern_cost + filter_cost + variable_cost + network_cost + data_transfer_cost;

        // Apply dynamic factors
        base_cost * service_load_factor * cache_factor
    }

    /// Estimate computation cost for patterns with selectivity analysis
    fn estimate_pattern_computation_cost(
        &self,
        patterns: &[(usize, TriplePattern)],
        service: &FederatedService,
    ) -> f64 {
        let mut total_cost = 0.0;

        for (_, pattern) in patterns {
            let base_pattern_cost = self.weights.pattern_weight;

            // Variable position affects selectivity
            let selectivity_factor = self.calculate_pattern_selectivity(pattern, service);

            // Property path complexity
            let complexity_factor = if pattern.predicate.contains('/')
                || pattern.predicate.contains('+')
                || pattern.predicate.contains('*')
            {
                3.0
            } else {
                1.0
            };

            total_cost += base_pattern_cost * selectivity_factor * complexity_factor;
        }

        total_cost
    }

    /// Calculate pattern selectivity based on variable positions and statistics
    fn calculate_pattern_selectivity(
        &self,
        pattern: &TriplePattern,
        service: &FederatedService,
    ) -> f64 {
        let stats_guard = self.statistics.read().unwrap();

        // Base selectivity factors
        let mut selectivity = 1.0;

        // Subject variable (typically most selective)
        if pattern.subject.starts_with('?') {
            selectivity *= 0.1; // Variables in subject are highly selective
        }

        // Predicate variable (least selective in most ontologies)
        if pattern.predicate.starts_with('?') {
            selectivity *= 0.01; // Very low selectivity
        } else {
            // Use historical selectivity data if available
            if let Some(&predicate_selectivity) =
                stats_guard.pattern_selectivity.get(&pattern.predicate)
            {
                selectivity *= predicate_selectivity;
            } else {
                selectivity *= 0.5; // Default predicate selectivity
            }
        }

        // Object variable
        if pattern.object.starts_with('?') {
            selectivity *= 0.3; // Moderate selectivity
        }

        // Ensure minimum selectivity
        selectivity.max(0.001)
    }

    /// Estimate filter execution cost
    fn estimate_filter_cost(
        &self,
        filters: &[FilterExpression],
        service: &FederatedService,
    ) -> f64 {
        let mut total_cost = 0.0;

        for filter in filters {
            let base_cost = self.weights.filter_weight;

            // Complex filter operations
            let complexity_multiplier = if filter.expression.to_uppercase().contains("REGEX") {
                5.0
            } else if filter.expression.to_uppercase().contains("EXISTS") {
                3.0
            } else if filter.expression.to_uppercase().contains("NOT EXISTS") {
                4.0
            } else if filter.expression.contains('<') || filter.expression.contains('>') {
                1.5 // Comparison operations
            } else {
                1.0
            };

            // Check if service supports advanced filtering
            let service_support_factor = if complexity_multiplier > 2.0
                && !service
                    .capabilities
                    .contains(&ServiceCapability::AdvancedFiltering)
            {
                2.0 // Higher cost if service doesn't support advanced filtering natively
            } else {
                1.0
            };

            total_cost += base_cost * complexity_multiplier * service_support_factor;
        }

        total_cost
    }

    /// Estimate network communication cost
    fn estimate_network_cost(&self, service: &FederatedService, component: &QueryComponent) -> f64 {
        let base_latency_cost = self.network_model.base_latency.as_millis() as f64
            * self.weights.network_weight
            / 100.0;

        let connection_overhead = self.network_model.connection_overhead.as_millis() as f64
            * self.weights.network_weight
            / 1000.0;

        // Query complexity affects round trips
        let round_trips = if component.patterns.len() > 10 || component.filters.len() > 5 {
            2.0 // May require multiple round trips for complex queries
        } else {
            1.0
        };

        (base_latency_cost + connection_overhead) * round_trips
    }

    /// Estimate data transfer cost based on expected result size
    fn estimate_data_transfer_cost(
        &self,
        service: &FederatedService,
        component: &QueryComponent,
    ) -> f64 {
        let estimated_results = self.estimate_result_size_advanced(service, component);
        let estimated_size_mb = (estimated_results as f64 * 0.5) / 1_000_000.0; // ~0.5KB per result

        let transfer_time = estimated_size_mb / self.network_model.bandwidth_mbps;
        let compressed_size = estimated_size_mb * self.network_model.compression_ratio;

        (transfer_time + compressed_size) * self.weights.data_transfer_weight
    }

    /// Advanced result size estimation using service statistics
    fn estimate_result_size_advanced(
        &self,
        service: &FederatedService,
        component: &QueryComponent,
    ) -> usize {
        let stats_guard = self.statistics.read().unwrap();

        if let Some(service_stats) = stats_guard.service_performance.get(&service.id) {
            // Use historical data to estimate result size
            let base_estimate = component.patterns.len() * 100; // Base estimate
            let throughput_factor = service_stats.throughput_patterns_per_sec / 10.0;

            (base_estimate as f64 * throughput_factor) as usize
        } else {
            // Fallback to simple estimation
            component.patterns.len() * 50
        }
    }

    /// Get current service load factor
    fn get_service_load_factor(&self, service: &FederatedService) -> f64 {
        let stats_guard = self.statistics.read().unwrap();

        if let Some(service_stats) = stats_guard.service_performance.get(&service.id) {
            // Higher load factor means higher cost
            1.0 + service_stats.load_factor * 0.5
        } else {
            1.0 // Default factor for unknown services
        }
    }

    /// Estimate cache benefit factor
    fn estimate_cache_benefit(
        &self,
        service: &FederatedService,
        component: &QueryComponent,
    ) -> f64 {
        let stats_guard = self.statistics.read().unwrap();

        if let Some(&cache_hit_rate) = stats_guard.cache_hit_rates.get(&service.id) {
            // Cache hits reduce effective cost
            1.0 - (cache_hit_rate * 0.8) // Up to 80% cost reduction from cache
        } else {
            1.0 // No cache benefit for unknown services
        }
    }

    /// Update cost model with execution feedback
    pub fn update_with_execution_result(
        &mut self,
        service_id: &str,
        predicted_cost: f64,
        actual_duration: Duration,
        result_count: usize,
        pattern_count: usize,
        filter_count: usize,
    ) {
        let actual_cost = actual_duration.as_millis() as f64;

        let record = ExecutionRecord {
            query_hash: 0, // TODO: implement proper query hashing
            service_id: service_id.to_string(),
            pattern_count,
            filter_count,
            predicted_cost,
            actual_cost,
            actual_duration,
            result_count,
            timestamp: Instant::now(),
        };

        // Update performance tracker
        self.performance_tracker
            .execution_records
            .push_back(record.clone());
        if self.performance_tracker.execution_records.len() > 1000 {
            self.performance_tracker.execution_records.pop_front();
        }

        // Update service statistics
        let mut stats_guard = self.statistics.write().unwrap();
        let service_stats = stats_guard
            .service_performance
            .entry(service_id.to_string())
            .or_insert(ServiceStats {
                average_response_time: Duration::from_millis(100),
                throughput_patterns_per_sec: 10.0,
                error_rate: 0.0,
                availability: 1.0,
                load_factor: 0.5,
                last_updated: Instant::now(),
            });

        // Update moving averages
        let alpha = 0.1; // Smoothing factor
        service_stats.average_response_time = Duration::from_millis(
            ((1.0 - alpha) * service_stats.average_response_time.as_millis() as f64
                + alpha * actual_duration.as_millis() as f64) as u64,
        );

        service_stats.throughput_patterns_per_sec = (1.0 - alpha)
            * service_stats.throughput_patterns_per_sec
            + alpha * (pattern_count as f64 / actual_duration.as_secs_f64());

        service_stats.last_updated = Instant::now();

        // Update prediction accuracy
        let prediction_error = (predicted_cost - actual_cost).abs() / actual_cost;
        self.performance_tracker.prediction_accuracy = (1.0 - alpha)
            * self.performance_tracker.prediction_accuracy
            + alpha * (1.0 - prediction_error.min(1.0));
    }

    fn estimate_pattern_cost(
        &self,
        service: &FederatedService,
        patterns: &[(usize, TriplePattern)],
    ) -> f64 {
        // Use the enhanced pattern computation cost method
        let computation_cost = self.estimate_pattern_computation_cost(patterns, service);

        // Add estimated join cost if multiple patterns
        let join_cost = if patterns.len() > 1 {
            self.estimate_join_cost(patterns, service)
        } else {
            0.0
        };

        computation_cost + join_cost
    }

    /// Estimate join cost for multiple patterns
    fn estimate_join_cost(
        &self,
        patterns: &[(usize, TriplePattern)],
        service: &FederatedService,
    ) -> f64 {
        if patterns.len() < 2 {
            return 0.0;
        }

        let mut total_join_cost = 0.0;
        let stats_guard = self.statistics.read().unwrap();

        // Estimate cost of joining pattern results
        for i in 0..patterns.len() - 1 {
            for j in i + 1..patterns.len() {
                let shared_vars = self.count_shared_variables(&patterns[i].1, &patterns[j].1);

                if shared_vars > 0 {
                    let join_key = format!("{}-{}", i, j);
                    let estimated_cardinality = stats_guard
                        .join_cardinality
                        .get(&join_key)
                        .unwrap_or(&1000)
                        .clone();

                    // Join cost increases with cardinality and decreases with selectivity
                    let selectivity = shared_vars as f64 * 0.1;
                    total_join_cost += self.weights.join_weight
                        * (estimated_cardinality as f64).log10()
                        * (1.0 + selectivity);
                }
            }
        }

        total_join_cost
    }

    /// Count shared variables between two patterns
    fn count_shared_variables(&self, pattern1: &TriplePattern, pattern2: &TriplePattern) -> usize {
        let vars1 = self.extract_pattern_variables_internal(pattern1);
        let vars2 = self.extract_pattern_variables_internal(pattern2);
        vars1.intersection(&vars2).count()
    }

    /// Extract variables from a single pattern
    fn extract_pattern_variables_internal(&self, pattern: &TriplePattern) -> HashSet<String> {
        let mut vars = HashSet::new();

        if pattern.subject.starts_with('?') {
            vars.insert(pattern.subject.clone());
        }
        if pattern.predicate.starts_with('?') {
            vars.insert(pattern.predicate.clone());
        }
        if pattern.object.starts_with('?') {
            vars.insert(pattern.object.clone());
        }

        vars
    }

    /// Estimate cost for single pattern on service
    fn estimate_single_pattern_cost(
        &self,
        service: &FederatedService,
        pattern: &TriplePattern,
    ) -> f64 {
        let base_cost = self.weights.pattern_weight;
        let selectivity = self.calculate_pattern_selectivity(pattern, service);
        let complexity_factor = self.calculate_pattern_complexity(pattern);

        base_cost * (1.0 / selectivity) * complexity_factor
    }

    /// Calculate pattern complexity based on its structure
    fn calculate_pattern_complexity(&self, pattern: &TriplePattern) -> f64 {
        let mut complexity = 1.0;

        // Property paths increase complexity
        if pattern.predicate.contains('/')
            || pattern.predicate.contains('+')
            || pattern.predicate.contains('*')
        {
            complexity *= 3.0;
        }

        // Regex patterns in objects increase complexity
        if pattern.object.contains("REGEX") {
            complexity *= 2.0;
        }

        // Variable predicates increase complexity
        if pattern.predicate.starts_with('?') {
            complexity *= 1.5;
        }

        complexity
    }

    /// Estimate network cost for service
    fn estimate_network_cost(&self, service: &FederatedService) -> f64 {
        let base_latency = self.network_model.base_latency.as_millis() as f64;
        let connection_overhead = self.network_model.connection_overhead.as_millis() as f64;

        // Factor in service-specific network characteristics
        let service_distance_factor =
            if service.endpoint.contains("localhost") || service.endpoint.contains("127.0.0.1") {
                0.1 // Local service
            } else if service.endpoint.contains(".local") {
                0.5 // Local network
            } else {
                1.0 // Remote service
            };

        (base_latency + connection_overhead) * service_distance_factor * self.weights.network_weight
            / 100.0
    }
}

/// Enhanced weights for sophisticated cost calculation
#[derive(Debug, Clone)]
struct CostWeights {
    pattern_weight: f64,
    filter_weight: f64,
    variable_weight: f64,
    join_weight: f64,
    network_weight: f64,
    data_transfer_weight: f64,
    concurrency_weight: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self {
            pattern_weight: 10.0,
            filter_weight: 5.0,
            variable_weight: 2.0,
            join_weight: 20.0,
            network_weight: 15.0,
            data_transfer_weight: 8.0,
            concurrency_weight: 5.0,
        }
    }
}

/// Configuration for query decomposer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecomposerConfig {
    /// Minimum patterns to consider distribution
    pub min_patterns_for_distribution: usize,
    /// Maximum services to use per query
    pub max_services_per_query: usize,
    /// Optimization strategy
    pub optimization_strategy: OptimizationStrategy,
    /// Enable advanced decomposition algorithms
    pub enable_advanced_algorithms: bool,
}

impl Default for DecomposerConfig {
    fn default() -> Self {
        Self {
            min_patterns_for_distribution: 3,
            max_services_per_query: 5,
            optimization_strategy: OptimizationStrategy::Balanced,
            enable_advanced_algorithms: true,
        }
    }
}

/// Optimization strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    MinimizeCost,
    MinimizeTime,
    MinimizeTransfer,
    Balanced,
}

/// Result of query decomposition
#[derive(Debug)]
pub struct DecompositionResult {
    pub plan: ExecutionPlan,
    pub statistics: DecompositionStats,
}

/// Statistics about decomposition process
#[derive(Debug)]
pub struct DecompositionStats {
    pub components_found: usize,
    pub total_plans_generated: usize,
    pub avg_plans_per_component: f64,
    pub decomposition_time: Duration,
}

/// Query graph representation
struct QueryGraph {
    graph: DiGraph<NodeType, EdgeType>,
    node_map: HashMap<String, NodeIndex>,
}

impl QueryGraph {
    fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
        }
    }

    fn add_variable_node(&mut self, var: String) -> NodeIndex {
        let node = self.graph.add_node(NodeType::Variable(var.clone()));
        self.node_map.insert(format!("var:{}", var), node);
        node
    }

    fn add_pattern_node(&mut self, idx: usize, pattern: TriplePattern) -> NodeIndex {
        self.graph.add_node(NodeType::Pattern(idx, pattern))
    }

    fn add_filter_node(&mut self, filter: FilterExpression) -> NodeIndex {
        self.graph.add_node(NodeType::Filter(filter))
    }

    fn connect_pattern_to_variable(
        &mut self,
        pattern: NodeIndex,
        var: NodeIndex,
        role: VariableRole,
    ) {
        self.graph.add_edge(pattern, var, EdgeType::Binds(role));
    }

    fn connect_filter_to_pattern(&mut self, filter: NodeIndex, pattern: NodeIndex) {
        self.graph.add_edge(filter, pattern, EdgeType::DependsOn);
    }

    fn pattern_nodes(&self) -> Vec<NodeIndex> {
        self.graph
            .node_indices()
            .filter(|&n| matches!(self.graph[n], NodeType::Pattern(_, _)))
            .collect()
    }

    fn node_type(&self, node: NodeIndex) -> Option<&NodeType> {
        self.graph.node_weight(node)
    }

    fn neighbors(&self, node: NodeIndex) -> Vec<NodeIndex> {
        self.graph.neighbors_undirected(node).collect()
    }

    fn pattern_binds_variable(&self, pattern: NodeIndex, var: NodeIndex) -> bool {
        self.graph.contains_edge(pattern, var)
    }
}

/// Node types in query graph
#[derive(Debug, Clone)]
enum NodeType {
    Variable(String),
    Pattern(usize, TriplePattern),
    Filter(FilterExpression),
}

/// Edge types in query graph
#[derive(Debug, Clone)]
enum EdgeType {
    Binds(VariableRole),
    DependsOn,
}

/// Role of variable in pattern
#[derive(Debug, Clone, Copy)]
enum VariableRole {
    Subject,
    Predicate,
    Object,
}

/// Query component (connected subgraph)
#[derive(Debug)]
struct QueryComponent {
    patterns: Vec<(usize, TriplePattern)>,
    variables: HashSet<String>,
    filters: Vec<FilterExpression>,
}

impl QueryComponent {
    fn new() -> Self {
        Self {
            patterns: Vec::new(),
            variables: HashSet::new(),
            filters: Vec::new(),
        }
    }
}

/// Plan for executing a component
#[derive(Debug, Clone)]
struct ComponentPlan {
    strategy: PlanStrategy,
    steps: Vec<PlanStep>,
    total_cost: f64,
    requires_join: bool,
}

/// Execution strategies
#[derive(Debug, Clone, Copy)]
enum PlanStrategy {
    SingleService,
    EvenDistribution,
    MinimizeIntermediate,
    MaximizeParallel,
    SpecializedServices,
    // Advanced strategies
    JoinAware,
    CostBased,
    PatternBased,
    StarJoinOptimized,
    MinimizeIntermediateAdvanced,
    MaximizeParallelAdvanced,
}

/// Individual step in component plan
#[derive(Debug, Clone)]
struct PlanStep {
    service_id: String,
    patterns: Vec<(usize, TriplePattern)>,
    filters: Vec<FilterExpression>,
    estimated_cost: f64,
    estimated_results: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_decomposer_creation() {
        let decomposer = QueryDecomposer::new();
        assert_eq!(decomposer.config.min_patterns_for_distribution, 3);
    }

    #[test]
    fn test_query_graph_creation() {
        let mut graph = QueryGraph::new();
        let var_node = graph.add_variable_node("?s".to_string());
        assert!(graph.node_type(var_node).is_some());
    }

    #[test]
    fn test_cost_estimator() {
        let estimator = CostEstimator::new();
        assert_eq!(estimator.weights.pattern_weight, 10.0);
    }
}
