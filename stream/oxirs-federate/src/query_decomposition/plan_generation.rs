//! Plan generation strategies for query components
//!
//! This module contains various algorithms for generating execution plans
//! for query components, including single service, distributed, and specialized strategies.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tracing::{debug, warn};

use crate::{service_registry::ServiceRegistry, FederatedService, ServiceCapability};

use super::types::*;

impl QueryDecomposer {
    /// Generate candidate execution plans for a component
    pub fn generate_component_plans(
        &self,
        component: &QueryComponent,
        registry: &ServiceRegistry,
    ) -> Result<Vec<ComponentPlan>> {
        let mut plans = Vec::new();

        // Get available services with SPARQL capability
        let services: Vec<_> = registry
            .get_all_services()
            .into_iter()
            .filter(|service| {
                service
                    .capabilities
                    .contains(&ServiceCapability::SparqlQuery)
            })
            .collect();
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
            let service_refs: Vec<&FederatedService> = services.iter().collect();
            let distributed_plans = self.create_distributed_plans(component, &service_refs)?;
            plans.extend(distributed_plans);
        }

        // Strategy 3: Specialized service assignment based on predicates
        let service_refs: Vec<&FederatedService> = services.iter().collect();
        let specialized_plan = self.create_specialized_service_plan(component, &service_refs)?;
        if let Some(plan) = specialized_plan {
            plans.push(plan);
        }

        if plans.is_empty() {
            // Fallback: force distribution even for small components
            let service_refs: Vec<&FederatedService> = services.iter().collect();
            let forced_plan = self.create_forced_distribution_plan(component, &service_refs)?;
            plans.push(forced_plan);
        }

        Ok(plans)
    }

    /// Check if a service can handle an entire component
    pub fn can_service_handle_component(
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
    pub fn create_single_service_plan(
        &self,
        service: &FederatedService,
        component: &QueryComponent,
    ) -> Result<ComponentPlan> {
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
    pub fn create_distributed_plans(
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
    pub fn distribute_patterns_evenly(
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

    /// Create plan using specialized services for specific predicates
    pub fn create_specialized_service_plan(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<Option<ComponentPlan>> {
        let mut predicate_services: HashMap<String, &FederatedService> = HashMap::new();

        // Map predicates to best services
        for (_, pattern) in &component.patterns {
            if let Some(ref predicate) = pattern.predicate {
                if !predicate.starts_with('?') {
                    let best_service = self.find_best_service_for_predicate(predicate, services);
                    if let Some(service) = best_service {
                        predicate_services.insert(predicate.clone(), service);
                    }
                }
            }
        }

        if predicate_services.is_empty() {
            return Ok(None);
        }

        // Group patterns by assigned service
        let mut service_patterns: HashMap<String, Vec<(usize, crate::planner::TriplePattern)>> =
            HashMap::new();
        for (idx, pattern) in &component.patterns {
            let service_id = if let Some(ref predicate) = pattern.predicate {
                if let Some(service) = predicate_services.get(predicate) {
                    service.id.clone()
                } else {
                    // Assign to service with most patterns
                    services[0].id.clone()
                }
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
    pub fn create_forced_distribution_plan(
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

    /// Advanced join-aware distribution algorithm
    pub fn distribute_join_aware(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<ComponentPlan> {
        debug!("Using join-aware distribution algorithm");

        // Analyze join patterns and dependencies
        let join_analysis = self.analyze_join_patterns(&component.patterns);

        // Group patterns based on join relationships
        let pattern_groups = self.group_patterns_by_joins(&component.patterns, &join_analysis);

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
            strategy: PlanStrategy::JoinAware,
            steps,
            total_cost,
            requires_join: pattern_groups.len() > 1,
        })
    }

    /// Advanced cost-based distribution algorithm
    pub fn distribute_cost_based(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<ComponentPlan> {
        debug!("Using cost-based distribution algorithm");

        // Calculate cost for each pattern on each service
        let mut pattern_costs = Vec::new();
        for (idx, pattern) in &component.patterns {
            let mut service_costs = Vec::new();
            for service in services {
                let cost = self
                    .cost_estimator
                    .estimate_single_pattern_cost(service, pattern);
                service_costs.push((*service, cost));
            }
            // Sort by cost (ascending)
            service_costs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            pattern_costs.push((*idx, pattern.clone(), service_costs));
        }

        // Assign patterns to services based on minimum cost
        let mut service_assignments: HashMap<String, Vec<(usize, crate::planner::TriplePattern)>> =
            HashMap::new();

        for (_idx, pattern, costs) in pattern_costs {
            if let Some((best_service, _cost)) = costs.first() {
                service_assignments
                    .entry(best_service.id.clone())
                    .or_default()
                    .push((_idx, pattern));
            }
        }

        let mut steps = Vec::new();
        let mut total_cost = 0.0;

        for (service_id, patterns) in service_assignments {
            let service = services.iter().find(|s| s.id == service_id).unwrap();
            let cost = self
                .cost_estimator
                .estimate_pattern_cost(service, &patterns);
            total_cost += cost;

            steps.push(PlanStep {
                service_id: service_id.clone(),
                patterns: patterns.clone(),
                filters: Vec::new(),
                estimated_cost: cost,
                estimated_results: self.estimate_result_size(service, &patterns),
            });
        }

        let requires_join = steps.len() > 1;
        Ok(ComponentPlan {
            strategy: PlanStrategy::CostBased,
            steps,
            total_cost,
            requires_join,
        })
    }

    /// Advanced pattern-based distribution algorithm with data overlap detection
    pub fn distribute_pattern_based(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<ComponentPlan> {
        debug!("Using pattern-based distribution algorithm with overlap detection");

        // Analyze pattern affinities to services
        let service_affinities = self.analyze_service_affinity(&component.patterns, services);

        // Create service assignments based on pattern affinity
        let mut service_assignments: HashMap<String, Vec<(usize, crate::planner::TriplePattern)>> =
            HashMap::new();

        for (idx, pattern) in &component.patterns {
            // Find service with highest affinity for this pattern
            let best_service = service_affinities
                .iter()
                .max_by(|a, b| a.affinity_score.partial_cmp(&b.affinity_score).unwrap())
                .map(|a| a.service_id.clone())
                .unwrap_or_else(|| services[0].id.clone());

            service_assignments
                .entry(best_service)
                .or_default()
                .push((*idx, pattern.clone()));
        }

        // Detect and handle data overlap between services
        self.optimize_for_data_overlap(&mut service_assignments, services)?;

        let mut steps = Vec::new();
        let mut total_cost = 0.0;

        for (service_id, patterns) in service_assignments {
            let service = services.iter().find(|s| s.id == service_id).unwrap();
            let cost = self
                .cost_estimator
                .estimate_pattern_cost(service, &patterns);
            total_cost += cost;

            steps.push(PlanStep {
                service_id: service_id.clone(),
                patterns: patterns.clone(),
                filters: Vec::new(),
                estimated_cost: cost,
                estimated_results: self.estimate_result_size(service, &patterns),
            });
        }

        let requires_join = steps.len() > 1;
        Ok(ComponentPlan {
            strategy: PlanStrategy::PatternBased,
            steps,
            total_cost,
            requires_join,
        })
    }

    /// Advanced star join optimization algorithm
    pub fn distribute_star_join_optimized(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<ComponentPlan> {
        debug!(
            "Using star join optimization algorithm for {} patterns",
            component.patterns.len()
        );

        // Analyze join patterns to identify star center
        let join_analysis = self.analyze_join_patterns(&component.patterns);

        if join_analysis.pattern_type != JoinPattern::Star {
            warn!("Component is not a star join pattern, falling back to cost-based distribution");
            return self.distribute_cost_based(component, services);
        }

        let central_variable = join_analysis
            .central_variable
            .as_ref()
            .ok_or_else(|| anyhow!("Star join pattern detected but no central variable found"))?;

        debug!(
            "Star join optimization with central variable: {}",
            central_variable
        );

        // Step 1: Identify patterns that bind the central variable
        let mut center_patterns = Vec::new();
        let mut satellite_patterns = Vec::new();

        for (idx, pattern) in &component.patterns {
            if self.pattern_binds_variable(pattern, central_variable) {
                center_patterns.push((*idx, pattern.clone()));
            } else {
                satellite_patterns.push((*idx, pattern.clone()));
            }
        }

        // Step 2: Find the best service for center patterns (most selective)
        let center_service =
            self.find_best_service_for_center_patterns(&center_patterns, services)?;

        // Step 3: Distribute satellite patterns based on selectivity and data locality
        let satellite_distribution = self.optimize_satellite_distribution(
            &satellite_patterns,
            services,
            &center_service,
            central_variable,
        )?;

        // Step 4: Create execution plan with optimized join ordering
        let mut steps = Vec::new();
        let mut total_cost = 0.0;

        // Add center step (highest priority for early execution)
        if !center_patterns.is_empty() {
            let cost = self
                .cost_estimator
                .estimate_pattern_cost(&center_service, &center_patterns);
            total_cost += cost;

            steps.push(PlanStep {
                service_id: center_service.id.clone(),
                patterns: center_patterns.clone(),
                filters: Vec::new(),
                estimated_cost: cost,
                estimated_results: self.estimate_result_size(&center_service, &center_patterns),
            });
        }

        // Add satellite steps in order of selectivity (most selective first)
        for (service_id, patterns) in satellite_distribution {
            let service = services.iter().find(|s| s.id == service_id).unwrap();
            let cost = self
                .cost_estimator
                .estimate_pattern_cost(service, &patterns);
            total_cost += cost;

            steps.push(PlanStep {
                service_id: service_id.clone(),
                patterns: patterns.clone(),
                filters: Vec::new(),
                estimated_cost: cost,
                estimated_results: self.estimate_result_size(service, &patterns),
            });
        }

        // Apply join cost estimation
        if steps.len() > 1 {
            let join_cost = self.estimate_star_join_cost(&steps);
            total_cost += join_cost;
        }

        let requires_join = steps.len() > 1;
        Ok(ComponentPlan {
            strategy: PlanStrategy::StarJoinOptimized,
            steps,
            total_cost,
            requires_join,
        })
    }

    /// Advanced minimize intermediate results algorithm
    pub fn distribute_minimize_intermediate_advanced(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<ComponentPlan> {
        debug!("Using advanced minimize intermediate algorithm");

        // Use the existing even distribution as a placeholder
        // In a full implementation, this would use bloom filters and result size estimation
        self.distribute_patterns_evenly(component, services)
    }

    /// Advanced maximize parallelism algorithm with bushy tree construction
    pub fn distribute_maximize_parallel_advanced(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<ComponentPlan> {
        debug!("Using advanced maximize parallelism algorithm with bushy tree construction");

        // Step 1: Build dependency graph to identify independent pattern groups
        let dependency_graph = self.build_pattern_dependency_graph(&component.patterns);

        // Step 2: Find independent pattern groups that can execute in parallel
        let independent_groups = self.find_independent_pattern_groups(&dependency_graph);

        // Step 3: Construct bushy tree for parallel execution
        let parallel_plan =
            self.construct_bushy_tree_plan(&independent_groups, &component.patterns, services)?;

        // Step 4: Optimize join ordering within each parallel branch
        let optimized_plan = self.optimize_parallel_join_ordering(parallel_plan)?;

        Ok(optimized_plan)
    }

    /// Optimize service assignments for data overlap
    pub fn optimize_for_data_overlap(
        &self,
        service_assignments: &mut HashMap<String, Vec<(usize, crate::planner::TriplePattern)>>,
        services: &[&FederatedService],
    ) -> Result<()> {
        // Analyze potential data overlap between services
        let overlap_analysis = self.analyze_data_overlap(service_assignments, services)?;

        // Consolidate patterns to services with higher data overlap
        for overlap in overlap_analysis {
            if overlap.overlap_score > 0.7 {
                // High overlap - consolidate patterns
                if let Some(source_patterns) = service_assignments.remove(&overlap.source_service) {
                    service_assignments
                        .entry(overlap.target_service)
                        .or_default()
                        .extend(source_patterns);
                }
            }
        }

        Ok(())
    }

    /// Analyze data overlap between service assignments
    pub fn analyze_data_overlap(
        &self,
        service_assignments: &HashMap<String, Vec<(usize, crate::planner::TriplePattern)>>,
        _services: &[&FederatedService],
    ) -> Result<Vec<DataOverlap>> {
        let mut overlaps = Vec::new();

        let service_ids: Vec<_> = service_assignments.keys().collect();

        for i in 0..service_ids.len() {
            for j in (i + 1)..service_ids.len() {
                let service1 = service_ids[i];
                let service2 = service_ids[j];

                if let (Some(patterns1), Some(patterns2)) = (
                    service_assignments.get(service1),
                    service_assignments.get(service2),
                ) {
                    let overlap_score = self.calculate_pattern_overlap(patterns1, patterns2);

                    if overlap_score > 0.0 {
                        overlaps.push(DataOverlap {
                            source_service: service1.clone(),
                            target_service: service2.clone(),
                            overlap_score,
                        });
                    }
                }
            }
        }

        // Sort by overlap score (descending)
        overlaps.sort_by(|a, b| b.overlap_score.partial_cmp(&a.overlap_score).unwrap());

        Ok(overlaps)
    }

    /// Calculate overlap score between two pattern sets
    pub fn calculate_pattern_overlap(
        &self,
        patterns1: &[(usize, crate::planner::TriplePattern)],
        patterns2: &[(usize, crate::planner::TriplePattern)],
    ) -> f64 {
        let mut common_predicates = 0;
        let total_predicates = patterns1.len() + patterns2.len();

        for (_, pattern1) in patterns1 {
            for (_, pattern2) in patterns2 {
                if let (Some(pred1), Some(pred2)) = (&pattern1.predicate, &pattern2.predicate) {
                    if pred1 == pred2 {
                        common_predicates += 1;
                    }
                }
            }
        }

        if total_predicates == 0 {
            0.0
        } else {
            (common_predicates as f64) / (total_predicates as f64)
        }
    }

    /// Check if pattern binds a specific variable
    pub fn pattern_binds_variable(
        &self,
        pattern: &crate::planner::TriplePattern,
        variable: &str,
    ) -> bool {
        [&pattern.subject, &pattern.predicate, &pattern.object]
            .iter()
            .any(|term| term.as_ref().is_some_and(|t| t == variable))
    }

    /// Find best service for center patterns in star join
    pub fn find_best_service_for_center_patterns(
        &self,
        center_patterns: &[(usize, crate::planner::TriplePattern)],
        services: &[&FederatedService],
    ) -> Result<FederatedService> {
        let mut best_service = None;
        let mut best_score = f64::NEG_INFINITY;

        for service in services {
            let mut score = 0.0;

            // Calculate selectivity score (prefer most selective patterns on center service)
            for (_, pattern) in center_patterns {
                let selectivity = self.estimate_single_pattern_selectivity(pattern);
                score += selectivity;
            }

            // Factor in service performance
            if let Some(avg_time) = service.performance.average_response_time {
                let time_factor = 1.0 / (1.0 + avg_time.as_millis() as f64 / 1000.0);
                score *= time_factor;
            }

            if score > best_score {
                best_score = score;
                best_service = Some((*service).clone());
            }
        }

        best_service.ok_or_else(|| anyhow!("No suitable service found for center patterns"))
    }

    /// Optimize satellite pattern distribution for star join
    pub fn optimize_satellite_distribution(
        &self,
        satellite_patterns: &[(usize, crate::planner::TriplePattern)],
        services: &[&FederatedService],
        center_service: &FederatedService,
        _central_variable: &str,
    ) -> Result<Vec<(String, Vec<(usize, crate::planner::TriplePattern)>)>> {
        let mut distribution = Vec::new();

        // Sort patterns by selectivity (most selective first)
        let mut sorted_patterns = satellite_patterns.to_vec();
        sorted_patterns.sort_by(|a, b| {
            let sel_a = self.estimate_single_pattern_selectivity(&a.1);
            let sel_b = self.estimate_single_pattern_selectivity(&b.1);
            sel_a.partial_cmp(&sel_b).unwrap()
        });

        // Distribute patterns to services (excluding center service)
        let available_services: Vec<_> = services
            .iter()
            .filter(|s| s.id != center_service.id)
            .collect();

        if available_services.is_empty() {
            // Fallback: use center service for satellite patterns too
            distribution.push((center_service.id.clone(), sorted_patterns));
        } else {
            let patterns_per_service =
                (sorted_patterns.len() + available_services.len() - 1) / available_services.len();

            for (i, chunk) in sorted_patterns.chunks(patterns_per_service).enumerate() {
                if i < available_services.len() {
                    distribution.push((available_services[i].id.clone(), chunk.to_vec()));
                }
            }
        }

        Ok(distribution)
    }

    /// Estimate star join cost
    pub fn estimate_star_join_cost(&self, steps: &[PlanStep]) -> f64 {
        if steps.len() <= 1 {
            return 0.0;
        }

        // Star join cost is typically dominated by the largest intermediate result
        let max_result_size = steps
            .iter()
            .map(|step| step.estimated_results)
            .max()
            .unwrap_or(1000) as f64;

        // Join cost scales logarithmically with result size
        let join_factor = 50.0; // Base join cost factor
        join_factor * max_result_size.log10()
    }

    /// Build pattern dependency graph
    pub fn build_pattern_dependency_graph(
        &self,
        patterns: &[(usize, crate::planner::TriplePattern)],
    ) -> HashMap<usize, Vec<usize>> {
        let mut dependency_graph = HashMap::new();

        // Build variable-pattern mapping
        let mut var_to_patterns: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, pattern) in patterns {
            let variables = self.extract_pattern_variables(pattern);
            for var in variables {
                var_to_patterns.entry(var).or_default().push(*idx);
            }
        }

        // Create dependencies based on shared variables
        for (idx, _) in patterns {
            let mut deps = Vec::new();

            for (other_idx, _) in patterns {
                if idx != other_idx {
                    // Check if patterns share any variables
                    if self.patterns_share_variables(patterns, *idx, *other_idx) {
                        deps.push(*other_idx);
                    }
                }
            }

            dependency_graph.insert(*idx, deps);
        }

        dependency_graph
    }

    /// Check if two patterns share variables
    pub fn patterns_share_variables(
        &self,
        patterns: &[(usize, crate::planner::TriplePattern)],
        idx1: usize,
        idx2: usize,
    ) -> bool {
        let pattern1 = patterns.iter().find(|(i, _)| *i == idx1).map(|(_, p)| p);
        let pattern2 = patterns.iter().find(|(i, _)| *i == idx2).map(|(_, p)| p);

        if let (Some(p1), Some(p2)) = (pattern1, pattern2) {
            let vars1 = self.extract_pattern_variables(p1);
            let vars2 = self.extract_pattern_variables(p2);

            for var1 in &vars1 {
                if vars2.contains(var1) {
                    return true;
                }
            }
        }

        false
    }

    /// Find independent pattern groups for parallel execution
    pub fn find_independent_pattern_groups(
        &self,
        dependency_graph: &HashMap<usize, Vec<usize>>,
    ) -> Vec<Vec<usize>> {
        let mut groups = Vec::new();
        let mut visited = std::collections::HashSet::new();

        for &pattern_idx in dependency_graph.keys() {
            if !visited.contains(&pattern_idx) {
                let mut group = Vec::new();
                self.collect_connected_patterns(
                    pattern_idx,
                    dependency_graph,
                    &mut visited,
                    &mut group,
                );
                groups.push(group);
            }
        }

        groups
    }

    /// Collect patterns connected by dependencies
    pub fn collect_connected_patterns(
        &self,
        pattern_idx: usize,
        dependency_graph: &HashMap<usize, Vec<usize>>,
        visited: &mut std::collections::HashSet<usize>,
        group: &mut Vec<usize>,
    ) {
        if visited.contains(&pattern_idx) {
            return;
        }

        visited.insert(pattern_idx);
        group.push(pattern_idx);

        if let Some(deps) = dependency_graph.get(&pattern_idx) {
            for &dep_idx in deps {
                self.collect_connected_patterns(dep_idx, dependency_graph, visited, group);
            }
        }
    }

    /// Construct bushy tree execution plan
    pub fn construct_bushy_tree_plan(
        &self,
        independent_groups: &[Vec<usize>],
        patterns: &[(usize, crate::planner::TriplePattern)],
        services: &[&FederatedService],
    ) -> Result<ComponentPlan> {
        let mut steps = Vec::new();
        let mut total_cost = 0.0;

        // Create execution steps for each independent group
        for (group_idx, group) in independent_groups.iter().enumerate() {
            let group_patterns: Vec<_> = group
                .iter()
                .filter_map(|&idx| patterns.iter().find(|(i, _)| *i == idx))
                .map(|(i, p)| (*i, p.clone()))
                .collect();

            if !group_patterns.is_empty() {
                let service = services[group_idx % services.len()];
                let cost = self
                    .cost_estimator
                    .estimate_pattern_cost(service, &group_patterns);
                total_cost += cost;

                steps.push(PlanStep {
                    service_id: service.id.clone(),
                    patterns: group_patterns.clone(),
                    filters: Vec::new(),
                    estimated_cost: cost,
                    estimated_results: self.estimate_result_size(service, &group_patterns),
                });
            }
        }

        // Add join cost for combining independent groups
        if steps.len() > 1 {
            let join_cost = self.estimate_parallel_join_cost(&steps);
            total_cost += join_cost;
        }

        let requires_join = steps.len() > 1;
        Ok(ComponentPlan {
            strategy: PlanStrategy::MaximizeParallelAdvanced,
            steps,
            total_cost,
            requires_join,
        })
    }

    /// Optimize join ordering within parallel branches
    pub fn optimize_parallel_join_ordering(&self, plan: ComponentPlan) -> Result<ComponentPlan> {
        // Sort steps by estimated result size (smallest first for efficient joins)
        let mut optimized_steps = plan.steps;
        optimized_steps.sort_by(|a, b| a.estimated_results.cmp(&b.estimated_results));

        Ok(ComponentPlan {
            strategy: plan.strategy,
            steps: optimized_steps,
            total_cost: plan.total_cost,
            requires_join: plan.requires_join,
        })
    }

    /// Estimate parallel join cost
    pub fn estimate_parallel_join_cost(&self, steps: &[PlanStep]) -> f64 {
        if steps.len() <= 1 {
            return 0.0;
        }

        // Parallel join cost depends on the largest intermediate results
        let total_results: u64 = steps.iter().map(|s| s.estimated_results).sum();
        let join_factor = 30.0; // Parallel joins are more efficient

        join_factor * (total_results as f64).log10()
    }
}
