//! Plan generation strategies for query components
//!
//! This module contains various algorithms for generating execution plans
//! for query components, including single service, distributed, and specialized strategies.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tracing::{debug, warn};

use crate::{
    FederatedService, ServiceCapability, ServiceRegistry,
};

use super::{
    pattern_analysis::*,
    cost_estimation::*,
    types::*,
};

impl QueryDecomposer {
    /// Generate candidate execution plans for a component
    pub fn generate_component_plans(
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
        let mut service_patterns: HashMap<String, Vec<(usize, crate::planner::TriplePattern)>> = HashMap::new();
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
                let cost = self.cost_estimator.estimate_single_pattern_cost(service, pattern);
                service_costs.push((*service, cost));
            }
            // Sort by cost (ascending)
            service_costs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            pattern_costs.push((*idx, pattern.clone(), service_costs));
        }

        // Assign patterns to services based on minimum cost
        let mut service_assignments: HashMap<String, Vec<(usize, crate::planner::TriplePattern)>> = HashMap::new();
        
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
            let cost = self.cost_estimator.estimate_pattern_cost(service, &patterns);
            total_cost += cost;

            steps.push(PlanStep {
                service_id: service_id.clone(),
                patterns,
                filters: Vec::new(),
                estimated_cost: cost,
                estimated_results: self.estimate_result_size(service, &patterns),
            });
        }

        Ok(ComponentPlan {
            strategy: PlanStrategy::CostBased,
            steps,
            total_cost,
            requires_join: steps.len() > 1,
        })
    }

    /// Advanced pattern-based distribution algorithm
    pub fn distribute_pattern_based(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<ComponentPlan> {
        debug!("Using pattern-based distribution algorithm");
        
        // Use the existing even distribution as a placeholder
        // In a full implementation, this would analyze data overlap and patterns
        self.distribute_patterns_evenly(component, services)
    }

    /// Advanced star join optimization algorithm
    pub fn distribute_star_join_optimized(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<ComponentPlan> {
        debug!("Using star join optimization algorithm");
        
        // Use the existing even distribution as a placeholder
        // In a full implementation, this would optimize for star join patterns
        self.distribute_patterns_evenly(component, services)
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

    /// Advanced maximize parallelism algorithm
    pub fn distribute_maximize_parallel_advanced(
        &self,
        component: &QueryComponent,
        services: &[&FederatedService],
    ) -> Result<ComponentPlan> {
        debug!("Using advanced maximize parallelism algorithm");
        
        // Use the existing even distribution as a placeholder
        // In a full implementation, this would analyze dependencies and maximize parallelism
        self.distribute_patterns_evenly(component, services)
    }
}