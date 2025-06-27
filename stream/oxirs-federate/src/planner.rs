//! Query Planning and Decomposition
//!
//! This module handles the analysis of queries and creates execution plans
//! for federated query processing across multiple services.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::{
    FederatedService, ServiceCapability, ServiceRegistry,
    query_decomposition::{QueryDecomposer, DecompositionResult, DecomposerConfig},
    service_optimizer::{ServiceOptimizer, ServiceOptimizerConfig},
};

/// Query planner for federated queries
#[derive(Debug)]
pub struct QueryPlanner {
    config: QueryPlannerConfig,
    decomposer: QueryDecomposer,
    service_optimizer: ServiceOptimizer,
}

impl QueryPlanner {
    /// Create a new query planner with default configuration
    pub fn new() -> Self {
        Self {
            config: QueryPlannerConfig::default(),
            decomposer: QueryDecomposer::new(),
            service_optimizer: ServiceOptimizer::new(),
        }
    }

    /// Create a new query planner with custom configuration
    pub fn with_config(config: QueryPlannerConfig) -> Self {
        let decomposer_config = DecomposerConfig {
            min_patterns_for_distribution: 3,
            max_services_per_query: config.max_services_per_query,
            optimization_strategy: match config.optimization_level {
                OptimizationLevel::None => crate::query_decomposition::OptimizationStrategy::MinimizeTime,
                OptimizationLevel::Basic => crate::query_decomposition::OptimizationStrategy::MinimizeCost,
                OptimizationLevel::Balanced => crate::query_decomposition::OptimizationStrategy::Balanced,
                OptimizationLevel::Aggressive => crate::query_decomposition::OptimizationStrategy::MinimizeTransfer,
            },
            enable_advanced_algorithms: config.optimization_level != OptimizationLevel::None,
        };
        
        let optimizer_config = ServiceOptimizerConfig {
            enable_pattern_grouping: config.optimization_level != OptimizationLevel::None,
            enable_service_merging: config.optimization_level == OptimizationLevel::Aggressive,
            enable_statistics: config.optimization_level != OptimizationLevel::None,
            ..Default::default()
        };
        
        Self {
            config,
            decomposer: QueryDecomposer::with_config(decomposer_config),
            service_optimizer: ServiceOptimizer::with_config(optimizer_config),
        }
    }

    /// Analyze a SPARQL query and extract planning information
    pub async fn analyze_sparql(&self, query: &str) -> Result<QueryInfo> {
        debug!("Analyzing SPARQL query: {}", query);

        let query_type = self.detect_query_type(query);
        let patterns = self.extract_triple_patterns(query)?;
        let service_clauses = self.extract_service_clauses(query)?;
        let filters = self.extract_filters(query)?;
        let variables = self.extract_variables(query)?;
        let complexity = self.calculate_complexity(&patterns, &filters, &service_clauses);

        let estimated_cost = self.estimate_query_cost(&patterns, &service_clauses);

        Ok(QueryInfo {
            query_type,
            original_query: query.to_string(),
            patterns,
            service_clauses,
            filters,
            variables,
            complexity,
            estimated_cost,
        })
    }

    /// Analyze a GraphQL query and extract planning information
    pub async fn analyze_graphql(
        &self,
        query: &str,
        variables: Option<&serde_json::Value>,
    ) -> Result<QueryInfo> {
        debug!("Analyzing GraphQL query: {}", query);

        let query_type = QueryType::GraphQLQuery;
        let selections = self.extract_graphql_selections(query)?;
        let _graphql_variables = variables.cloned().unwrap_or(serde_json::Value::Null);

        let patterns = self.graphql_to_patterns(&selections)?;
        let complexity = self.calculate_graphql_complexity(&selections);

        Ok(QueryInfo {
            query_type,
            original_query: query.to_string(),
            patterns,
            service_clauses: Vec::new(),
            filters: Vec::new(),
            variables: HashSet::new(),
            complexity,
            estimated_cost: self.estimate_graphql_cost(&selections),
        })
    }

    /// Create an execution plan for a SPARQL query
    pub async fn plan_sparql(
        &self,
        query_info: &QueryInfo,
        registry: &ServiceRegistry,
    ) -> Result<ExecutionPlan> {
        info!(
            "Planning SPARQL execution for {} patterns",
            query_info.patterns.len()
        );

        let mut plan = ExecutionPlan {
            query_id: uuid::Uuid::new_v4().to_string(),
            query_type: query_info.query_type,
            steps: Vec::new(),
            estimated_duration: Duration::from_secs(0),
            parallelizable_steps: Vec::new(),
            dependencies: HashMap::new(),
        };

        // Handle explicit SERVICE clauses with optimization
        if !query_info.service_clauses.is_empty() {
            debug!("Optimizing {} SERVICE clauses", query_info.service_clauses.len());
            
            // Optimize SERVICE clauses
            let optimized = self.service_optimizer.optimize_query(query_info, registry).await?;
            
            // Create steps from optimized services
            for opt_service in optimized.services {
                let step = ExecutionStep {
                    step_id: uuid::Uuid::new_v4().to_string(),
                    step_type: StepType::ServiceQuery,
                    service_id: Some(opt_service.service_id),
                    query_fragment: self.build_optimized_service_query(&opt_service),
                    expected_variables: self.extract_service_variables(&opt_service),
                    estimated_duration: Duration::from_millis(opt_service.estimated_cost as u64),
                    dependencies: Vec::new(),
                    parallel_group: if opt_service.strategy.stream_results { None } else { Some(0) },
                };
                plan.steps.push(step);
            }
            
            // Handle remaining global filters after optimization
            for filter in optimized.global_filters {
                let filter_step = ExecutionStep {
                    step_id: uuid::Uuid::new_v4().to_string(),
                    step_type: StepType::Filter,
                    service_id: None,
                    query_fragment: format!("FILTER({})", filter.expression),
                    expected_variables: filter.variables,
                    estimated_duration: Duration::from_millis(10),
                    dependencies: plan.steps.iter().map(|s| s.step_id.clone()).collect(),
                    parallel_group: None,
                };
                plan.steps.push(filter_step);
            }
        } else {
            // Handle queries without explicit SERVICE clauses
            for service_clause in &query_info.service_clauses {
                let step = self.create_service_step(service_clause, registry)?;
                plan.steps.push(step);
            }
        }

        // Group remaining patterns by compatible services
        let remaining_patterns: Vec<_> = query_info
            .patterns
            .iter()
            .filter(|pattern| {
                !self.is_pattern_in_service_clause(pattern, &query_info.service_clauses)
            })
            .collect();

        if !remaining_patterns.is_empty() {
            let service_assignments =
                self.assign_patterns_to_services(&remaining_patterns, registry)?;

            for (service_id, patterns) in service_assignments {
                let step = ExecutionStep {
                    step_id: uuid::Uuid::new_v4().to_string(),
                    step_type: StepType::ServiceQuery,
                    service_id: Some(service_id),
                    query_fragment: self.build_sparql_fragment(&patterns),
                    expected_variables: self.extract_pattern_variables(&patterns),
                    estimated_duration: Duration::from_millis(100),
                    dependencies: Vec::new(),
                    parallel_group: None,
                };
                plan.steps.push(step);
            }
        }

        // Add join steps if multiple services are involved
        if plan.steps.len() > 1 {
            let join_step = ExecutionStep {
                step_id: uuid::Uuid::new_v4().to_string(),
                step_type: StepType::Join,
                service_id: None,
                query_fragment: "-- Join results from multiple services".to_string(),
                expected_variables: query_info.variables.clone(),
                estimated_duration: Duration::from_millis(50),
                dependencies: plan.steps.iter().map(|s| s.step_id.clone()).collect(),
                parallel_group: None,
            };
            plan.steps.push(join_step);
        }

        self.optimize_plan(&mut plan);
        Ok(plan)
    }
    
    /// Create an execution plan using advanced decomposition for complex queries
    pub async fn plan_sparql_advanced(
        &self,
        query_info: &QueryInfo,
        registry: &ServiceRegistry,
    ) -> Result<ExecutionPlan> {
        info!("Using advanced query decomposition for {} patterns", query_info.patterns.len());
        
        // Use advanced decomposition for complex queries
        if query_info.complexity as u8 >= QueryComplexity::High as u8 || 
           query_info.patterns.len() >= self.config.advanced_decomposition_threshold {
            let decomposition_result = self.decomposer.decompose(query_info, registry).await?;
            
            info!(
                "Advanced decomposition found {} components, generated {} total plans",
                decomposition_result.statistics.components_found,
                decomposition_result.statistics.total_plans_generated
            );
            
            return Ok(decomposition_result.plan);
        }
        
        // Fall back to regular planning for simple queries
        self.plan_sparql(query_info, registry).await
    }

    /// Create an execution plan for a GraphQL query
    pub async fn plan_graphql(
        &self,
        query_info: &QueryInfo,
        registry: &ServiceRegistry,
    ) -> Result<ExecutionPlan> {
        info!("Planning GraphQL execution");

        let mut plan = ExecutionPlan {
            query_id: uuid::Uuid::new_v4().to_string(),
            query_type: query_info.query_type,
            steps: Vec::new(),
            estimated_duration: Duration::from_secs(0),
            parallelizable_steps: Vec::new(),
            dependencies: HashMap::new(),
        };

        let graphql_services: Vec<_> =
            registry.get_services_with_capability(&ServiceCapability::GraphQLQuery);

        if graphql_services.is_empty() {
            return Err(anyhow!("No GraphQL services available for federation"));
        }

        for service in graphql_services {
            let step = ExecutionStep {
                step_id: uuid::Uuid::new_v4().to_string(),
                step_type: StepType::GraphQLQuery,
                service_id: Some(service.id.clone()),
                query_fragment: query_info.original_query.clone(),
                expected_variables: HashSet::new(),
                estimated_duration: Duration::from_millis(200),
                dependencies: Vec::new(),
                parallel_group: Some(0),
            };
            plan.steps.push(step);
        }

        if plan.steps.len() > 1 {
            let stitch_step = ExecutionStep {
                step_id: uuid::Uuid::new_v4().to_string(),
                step_type: StepType::SchemaStitch,
                service_id: None,
                query_fragment: "-- Stitch GraphQL schemas".to_string(),
                expected_variables: HashSet::new(),
                estimated_duration: Duration::from_millis(30),
                dependencies: plan.steps.iter().map(|s| s.step_id.clone()).collect(),
                parallel_group: None,
            };
            plan.steps.push(stitch_step);
        }

        self.optimize_plan(&mut plan);
        Ok(plan)
    }

    fn detect_query_type(&self, query: &str) -> QueryType {
        let query_upper = query.to_uppercase();

        if query_upper.trim_start().starts_with("SELECT") {
            QueryType::SparqlSelect
        } else if query_upper.trim_start().starts_with("CONSTRUCT") {
            QueryType::SparqlConstruct
        } else if query_upper.trim_start().starts_with("ASK") {
            QueryType::SparqlAsk
        } else if query_upper.trim_start().starts_with("DESCRIBE") {
            QueryType::SparqlDescribe
        } else if query_upper.trim_start().starts_with("INSERT")
            || query_upper.trim_start().starts_with("DELETE")
        {
            QueryType::SparqlUpdate
        } else if query_upper.trim_start().starts_with("QUERY") || query_upper.contains("{") {
            QueryType::GraphQLQuery
        } else {
            QueryType::SparqlSelect
        }
    }

    fn extract_triple_patterns(&self, query: &str) -> Result<Vec<TriplePattern>> {
        let mut patterns = Vec::new();

        if let Some(where_start) = query.to_uppercase().find("WHERE") {
            let where_clause = &query[where_start + 5..];

            if let Some(open_brace) = where_clause.find('{') {
                if let Some(close_brace) = where_clause.rfind('}') {
                    let pattern_content = &where_clause[open_brace + 1..close_brace];

                    for line in pattern_content.split('.') {
                        let line = line.trim();
                        if line.is_empty() || line.starts_with("#") {
                            continue;
                        }

                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 3 {
                            patterns.push(TriplePattern {
                                subject: parts[0].to_string(),
                                predicate: parts[1].to_string(),
                                object: parts[2..].join(" "),
                                pattern_string: line.to_string(),
                            });
                        }
                    }
                }
            }
        }

        Ok(patterns)
    }

    fn extract_service_clauses(&self, query: &str) -> Result<Vec<ServiceClause>> {
        let mut services = Vec::new();
        let query_upper = query.to_uppercase();

        let mut start_pos = 0;
        while let Some(service_pos) = query_upper[start_pos..].find("SERVICE") {
            let actual_pos = start_pos + service_pos;

            if let Some(url_start) = query[actual_pos..].find('<') {
                if let Some(url_end) = query[actual_pos + url_start..].find('>') {
                    let service_url = query
                        [actual_pos + url_start + 1..actual_pos + url_start + url_end]
                        .to_string();

                    if let Some(brace_start) = query[actual_pos..].find('{') {
                        if let Some(brace_end) = query[actual_pos + brace_start..].find('}') {
                            let subquery = query[actual_pos + brace_start + 1
                                ..actual_pos + brace_start + brace_end]
                                .trim()
                                .to_string();

                            services.push(ServiceClause {
                                service_url,
                                subquery,
                                silent: query_upper[actual_pos..].starts_with("SERVICE SILENT"),
                            });
                        }
                    }
                }
            }

            start_pos = actual_pos + 7;
        }

        Ok(services)
    }

    fn extract_filters(&self, query: &str) -> Result<Vec<FilterExpression>> {
        let mut filters = Vec::new();
        let query_upper = query.to_uppercase();

        let mut start_pos = 0;
        while let Some(filter_pos) = query_upper[start_pos..].find("FILTER") {
            let actual_pos = start_pos + filter_pos;

            if let Some(paren_start) = query[actual_pos..].find('(') {
                let mut paren_count = 0;
                let mut end_pos = actual_pos + paren_start;

                for (i, char) in query[actual_pos + paren_start..].char_indices() {
                    if char == '(' {
                        paren_count += 1;
                    } else if char == ')' {
                        paren_count -= 1;
                        if paren_count == 0 {
                            end_pos = actual_pos + paren_start + i + 1;
                            break;
                        }
                    }
                }

                let expression = query[actual_pos + paren_start + 1..end_pos - 1]
                    .trim()
                    .to_string();
                let variables = self.extract_variables_from_text(&expression);

                filters.push(FilterExpression {
                    expression,
                    variables,
                });
            }

            start_pos = actual_pos + 6;
        }

        Ok(filters)
    }

    fn extract_variables(&self, query: &str) -> Result<HashSet<String>> {
        let mut variables = HashSet::new();

        if let Some(select_start) = query.to_uppercase().find("SELECT") {
            if let Some(where_start) = query.to_uppercase().find("WHERE") {
                let select_clause = &query[select_start + 6..where_start];

                for word in select_clause.split_whitespace() {
                    if word.starts_with('?') {
                        variables.insert(word.to_string());
                    }
                }

                if select_clause.trim() == "*" {
                    variables.extend(self.extract_variables_from_text(&query[where_start..]));
                }
            }
        }

        if let Some(where_start) = query.to_uppercase().find("WHERE") {
            variables.extend(self.extract_variables_from_text(&query[where_start..]));
        }

        Ok(variables)
    }

    fn extract_variables_from_text(&self, text: &str) -> HashSet<String> {
        let mut variables = HashSet::new();

        for word in text.split_whitespace() {
            if word.starts_with('?') {
                let clean_var = word.trim_end_matches(&['.', ';', '}', ')', ','][..]);
                variables.insert(clean_var.to_string());
            }
        }

        variables
    }

    fn extract_graphql_selections(&self, _query: &str) -> Result<Vec<GraphQLSelection>> {
        Ok(vec![GraphQLSelection {
            name: "placeholder".to_string(),
            arguments: HashMap::new(),
            selections: Vec::new(),
        }])
    }

    fn graphql_to_patterns(&self, _selections: &[GraphQLSelection]) -> Result<Vec<TriplePattern>> {
        Ok(Vec::new())
    }

    fn calculate_complexity(
        &self,
        patterns: &[TriplePattern],
        filters: &[FilterExpression],
        services: &[ServiceClause],
    ) -> QueryComplexity {
        let base_complexity = patterns.len() + filters.len() * 2 + services.len() * 3;

        if base_complexity < 5 {
            QueryComplexity::Low
        } else if base_complexity < 15 {
            QueryComplexity::Medium
        } else if base_complexity < 30 {
            QueryComplexity::High
        } else {
            QueryComplexity::VeryHigh
        }
    }

    fn calculate_graphql_complexity(&self, _selections: &[GraphQLSelection]) -> QueryComplexity {
        QueryComplexity::Medium
    }

    fn estimate_query_cost(&self, patterns: &[TriplePattern], services: &[ServiceClause]) -> u64 {
        (patterns.len() * 10 + services.len() * 50) as u64
    }

    fn estimate_graphql_cost(&self, _selections: &[GraphQLSelection]) -> u64 {
        100
    }

    fn create_service_step(
        &self,
        service_clause: &ServiceClause,
        registry: &ServiceRegistry,
    ) -> Result<ExecutionStep> {
        let service = registry
            .get_all_services()
            .find(|s| s.endpoint == service_clause.service_url)
            .ok_or_else(|| anyhow!("Service not found: {}", service_clause.service_url))?;

        Ok(ExecutionStep {
            step_id: uuid::Uuid::new_v4().to_string(),
            step_type: StepType::ServiceQuery,
            service_id: Some(service.id.clone()),
            query_fragment: service_clause.subquery.clone(),
            expected_variables: HashSet::new(),
            estimated_duration: Duration::from_millis(150),
            dependencies: Vec::new(),
            parallel_group: None,
        })
    }

    fn assign_patterns_to_services(
        &self,
        patterns: &[&TriplePattern],
        registry: &ServiceRegistry,
    ) -> Result<HashMap<String, Vec<TriplePattern>>> {
        let mut assignments = HashMap::new();

        let sparql_services: Vec<_> =
            registry.get_services_with_capability(&ServiceCapability::SparqlQuery);

        if sparql_services.is_empty() {
            return Err(anyhow!("No SPARQL services available"));
        }

        // Use different assignment strategies based on configuration
        match self.config.service_selection_strategy {
            ServiceSelectionStrategy::RoundRobin => {
                self.assign_patterns_round_robin(patterns, &sparql_services, &mut assignments)?;
            }
            ServiceSelectionStrategy::CapabilityBased => {
                self.assign_patterns_by_capability(patterns, &sparql_services, registry, &mut assignments)?;
            }
            ServiceSelectionStrategy::LoadBased => {
                self.assign_patterns_by_load(patterns, &sparql_services, registry, &mut assignments)?;
            }
            ServiceSelectionStrategy::CostBased => {
                self.assign_patterns_by_cost(patterns, &sparql_services, registry, &mut assignments)?;
            }
            ServiceSelectionStrategy::PredicateBased => {
                self.assign_patterns_by_predicate(patterns, &sparql_services, registry, &mut assignments)?;
            }
            ServiceSelectionStrategy::First => {
                // Original simple strategy - assign all to first service
                if let Some(service) = sparql_services.first() {
                    assignments.insert(
                        service.id.clone(),
                        patterns.iter().map(|p| (*p).clone()).collect(),
                    );
                }
            }
        }

        Ok(assignments)
    }

    /// Assign patterns using round-robin strategy
    fn assign_patterns_round_robin(
        &self,
        patterns: &[&TriplePattern],
        services: &[crate::FederatedService],
        assignments: &mut HashMap<String, Vec<TriplePattern>>,
    ) -> Result<()> {
        for (i, pattern) in patterns.iter().enumerate() {
            let service = &services[i % services.len()];
            assignments
                .entry(service.id.clone())
                .or_default()
                .push((*pattern).clone());
        }
        Ok(())
    }

    /// Assign patterns based on service capabilities and data coverage
    fn assign_patterns_by_capability(
        &self,
        patterns: &[&TriplePattern],
        services: &[crate::FederatedService],
        registry: &ServiceRegistry,
        assignments: &mut HashMap<String, Vec<TriplePattern>>,
    ) -> Result<()> {
        for pattern in patterns {
            let mut best_service = None;
            let mut best_score = 0.0;

            for service in services {
                let score = self.calculate_service_capability_score(pattern, service, registry);
                if score > best_score {
                    best_score = score;
                    best_service = Some(service);
                }
            }

            if let Some(service) = best_service {
                assignments
                    .entry(service.id.clone())
                    .or_default()
                    .push((*pattern).clone());
            } else {
                // Fallback to first service
                assignments
                    .entry(services[0].id.clone())
                    .or_default()
                    .push((*pattern).clone());
            }
        }
        Ok(())
    }

    /// Assign patterns based on current service load
    fn assign_patterns_by_load(
        &self,
        patterns: &[&TriplePattern],
        services: &[crate::FederatedService],
        registry: &ServiceRegistry,
        assignments: &mut HashMap<String, Vec<TriplePattern>>,
    ) -> Result<()> {
        for pattern in patterns {
            let mut best_service = None;
            let mut lowest_load = f64::INFINITY;

            for service in services {
                // Get current load from service status
                let load = self.get_service_load(service, registry);
                if load < lowest_load {
                    lowest_load = load;
                    best_service = Some(service);
                }
            }

            if let Some(service) = best_service {
                assignments
                    .entry(service.id.clone())
                    .or_default()
                    .push((*pattern).clone());
            }
        }
        Ok(())
    }

    /// Assign patterns based on estimated cost
    fn assign_patterns_by_cost(
        &self,
        patterns: &[&TriplePattern],
        services: &[crate::FederatedService],
        registry: &ServiceRegistry,
        assignments: &mut HashMap<String, Vec<TriplePattern>>,
    ) -> Result<()> {
        for pattern in patterns {
            let mut best_service = None;
            let mut lowest_cost = u64::MAX;

            for service in services {
                let cost = self.estimate_pattern_cost(pattern, service, registry);
                if cost < lowest_cost {
                    lowest_cost = cost;
                    best_service = Some(service);
                }
            }

            if let Some(service) = best_service {
                assignments
                    .entry(service.id.clone())
                    .or_default()
                    .push((*pattern).clone());
            }
        }
        Ok(())
    }

    /// Assign patterns based on predicate specialization
    fn assign_patterns_by_predicate(
        &self,
        patterns: &[&TriplePattern],
        services: &[crate::FederatedService],
        registry: &ServiceRegistry,
        assignments: &mut HashMap<String, Vec<TriplePattern>>,
    ) -> Result<()> {
        // Group patterns by predicate
        let mut predicate_groups: HashMap<String, Vec<&TriplePattern>> = HashMap::new();
        for pattern in patterns {
            predicate_groups
                .entry(pattern.predicate.clone())
                .or_default()
                .push(pattern);
        }

        // Assign each predicate group to the most suitable service
        for (predicate, predicate_patterns) in predicate_groups {
            let mut best_service = None;
            let mut best_score = 0.0;

            for service in services {
                let score = self.calculate_predicate_affinity_score(&predicate, service, registry);
                if score > best_score {
                    best_score = score;
                    best_service = Some(service);
                }
            }

            let target_service = best_service.unwrap_or(&services[0]);
            for pattern in predicate_patterns {
                assignments
                    .entry(target_service.id.clone())
                    .or_default()
                    .push((*pattern).clone());
            }
        }
        Ok(())
    }

    /// Calculate service capability score for a pattern
    fn calculate_service_capability_score(
        &self,
        pattern: &TriplePattern,
        service: &crate::FederatedService,
        _registry: &ServiceRegistry,
    ) -> f64 {
        let mut score = 0.0;

        // Basic capability scoring
        if service.capabilities.contains(&ServiceCapability::SparqlQuery) {
            score += 1.0;
        }
        if service.capabilities.contains(&ServiceCapability::FullTextSearch) && 
           pattern.pattern_string.contains("REGEX") {
            score += 0.5;
        }
        if service.capabilities.contains(&ServiceCapability::Geospatial) && 
           pattern.predicate.contains("geo:") {
            score += 0.5;
        }

        // Pattern complexity factor
        let complexity = self.calculate_pattern_complexity(pattern);
        score += match complexity {
            PatternComplexity::Simple => 0.1,
            PatternComplexity::Medium => 0.0,
            PatternComplexity::Complex => -0.1,
        };

        score
    }

    /// Get current load of a service
    fn get_service_load(&self, service: &crate::FederatedService, _registry: &ServiceRegistry) -> f64 {
        // In a real implementation, this would query service metrics
        // For now, use a simple heuristic based on performance characteristics
        if let Some(avg_response_time) = service.performance.average_response_time {
            // Convert response time to load estimate (0.0 to 1.0)
            let millis = avg_response_time.as_millis() as f64;
            (millis / 1000.0).min(1.0) // Normalize to 0-1 range
        } else {
            0.5 // Default moderate load
        }
    }

    /// Estimate cost of executing a pattern on a service
    fn estimate_pattern_cost(
        &self,
        pattern: &TriplePattern,
        service: &crate::FederatedService,
        _registry: &ServiceRegistry,
    ) -> u64 {
        let mut cost = 100; // Base cost

        // Pattern complexity factor
        let complexity = self.calculate_pattern_complexity(pattern);
        cost += match complexity {
            PatternComplexity::Simple => 10,
            PatternComplexity::Medium => 50,
            PatternComplexity::Complex => 200,
        };

        // Service performance factor based on average response time
        if let Some(avg_response_time) = service.performance.average_response_time {
            let perf_factor = avg_response_time.as_millis() as f64 / 100.0;
            cost = (cost as f64 * perf_factor.max(0.5)) as u64;
        }

        cost
    }

    /// Calculate predicate affinity score for a service
    fn calculate_predicate_affinity_score(
        &self,
        predicate: &str,
        service: &crate::FederatedService,
        _registry: &ServiceRegistry,
    ) -> f64 {
        let mut score = 1.0; // Base score

        // Check if service has specialized capabilities for this predicate
        if predicate.contains("geo:") && service.capabilities.contains(&ServiceCapability::Geospatial) {
            score += 2.0;
        }
        if predicate.contains("foaf:") && service.name.to_lowercase().contains("foaf") {
            score += 1.5;
        }
        if predicate.contains("dbo:") && service.name.to_lowercase().contains("dbpedia") {
            score += 1.5;
        }
        if predicate.contains("schema:") && service.name.to_lowercase().contains("schema") {
            score += 1.5;
        }

        // Performance factor based on response time (lower is better)
        if let Some(avg_response_time) = service.performance.average_response_time {
            let perf_factor = 100.0 / (avg_response_time.as_millis() as f64 + 1.0);
            score *= perf_factor;
        }

        score
    }

    /// Calculate pattern complexity
    fn calculate_pattern_complexity(&self, pattern: &TriplePattern) -> PatternComplexity {
        let variable_count = [&pattern.subject, &pattern.predicate, &pattern.object]
            .iter()
            .filter(|part| part.starts_with('?'))
            .count();

        if variable_count == 0 {
            PatternComplexity::Simple
        } else if variable_count <= 2 {
            PatternComplexity::Medium
        } else {
            PatternComplexity::Complex
        }
    }

    fn is_pattern_in_service_clause(
        &self,
        _pattern: &TriplePattern,
        _service_clauses: &[ServiceClause],
    ) -> bool {
        false
    }

    fn build_sparql_fragment(&self, patterns: &[TriplePattern]) -> String {
        let pattern_strings: Vec<String> =
            patterns.iter().map(|p| p.pattern_string.clone()).collect();

        format!(
            "SELECT * WHERE {{\n  {}\n}}",
            pattern_strings.join(" .\n  ")
        )
    }

    fn extract_pattern_variables(&self, patterns: &[TriplePattern]) -> HashSet<String> {
        let mut variables = HashSet::new();

        for pattern in patterns {
            if pattern.subject.starts_with('?') {
                variables.insert(pattern.subject.clone());
            }
            if pattern.predicate.starts_with('?') {
                variables.insert(pattern.predicate.clone());
            }
            if pattern.object.starts_with('?') {
                variables.insert(pattern.object.clone());
            }
        }

        variables
    }
    
    /// Build query from optimized service clause
    fn build_optimized_service_query(&self, service: &crate::service_optimizer::OptimizedServiceClause) -> String {
        let mut query = String::from("SELECT * WHERE {\n");
        
        // Add patterns
        for pattern in &service.patterns {
            query.push_str("  ");
            query.push_str(&pattern.pattern_string);
            query.push_str(" .\n");
        }
        
        // Add filters (both local and pushed)
        for filter in &service.filters {
            query.push_str("  FILTER(");
            query.push_str(&filter.expression);
            query.push_str(")\n");
        }
        
        query.push_str("}");
        
        // Add LIMIT if using batch processing
        if service.strategy.use_values_binding && service.strategy.batch_size > 0 {
            query.push_str(&format!(" LIMIT {}", service.strategy.batch_size));
        }
        
        query
    }
    
    /// Extract variables from optimized service clause
    fn extract_service_variables(&self, service: &crate::service_optimizer::OptimizedServiceClause) -> HashSet<String> {
        let mut vars = HashSet::new();
        
        for pattern in &service.patterns {
            if pattern.subject.starts_with('?') {
                vars.insert(pattern.subject.clone());
            }
            if pattern.predicate.starts_with('?') {
                vars.insert(pattern.predicate.clone());
            }
            if pattern.object.starts_with('?') {
                vars.insert(pattern.object.clone());
            }
        }
        
        for filter in &service.filters {
            vars.extend(filter.variables.iter().cloned());
        }
        
        vars
    }

    fn optimize_plan(&self, plan: &mut ExecutionPlan) {
        let mut parallel_groups = HashMap::new();
        let mut group_id = 0;

        for step in &mut plan.steps {
            if step.dependencies.is_empty() && step.service_id.is_some() {
                step.parallel_group = Some(group_id);
                parallel_groups
                    .entry(group_id)
                    .or_insert_with(Vec::new)
                    .push(step.step_id.clone());
            }
        }

        plan.parallelizable_steps = parallel_groups.into_values().collect();

        let mut max_parallel_duration = Duration::from_secs(0);
        let mut sequential_duration = Duration::from_secs(0);

        for step in &plan.steps {
            if step.parallel_group.is_some() {
                max_parallel_duration = max_parallel_duration.max(step.estimated_duration);
            } else {
                sequential_duration += step.estimated_duration;
            }
        }

        plan.estimated_duration = max_parallel_duration + sequential_duration;
    }
}

impl Default for QueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the query planner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlannerConfig {
    pub max_services_per_query: usize,
    pub optimization_level: OptimizationLevel,
    pub timeout: Duration,
    pub enable_caching: bool,
    pub cost_threshold: u64,
    pub service_selection_strategy: ServiceSelectionStrategy,
    pub advanced_decomposition_threshold: usize,
}

impl Default for QueryPlannerConfig {
    fn default() -> Self {
        Self {
            max_services_per_query: 10,
            optimization_level: OptimizationLevel::Balanced,
            timeout: Duration::from_secs(30),
            enable_caching: true,
            cost_threshold: 1000,
            service_selection_strategy: ServiceSelectionStrategy::CapabilityBased,
            advanced_decomposition_threshold: 5,
        }
    }
}

/// Service selection strategies for query planning
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ServiceSelectionStrategy {
    /// Assign patterns to first available service
    First,
    /// Round-robin assignment across services
    RoundRobin,
    /// Select services based on capabilities and pattern characteristics
    CapabilityBased,
    /// Select services based on current load
    LoadBased,
    /// Select services based on estimated execution cost
    CostBased,
    /// Group patterns by predicate and assign to specialized services
    PredicateBased,
}

/// Pattern complexity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PatternComplexity {
    Simple,
    Medium,
    Complex,
}

/// Optimization levels for query planning
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizationLevel {
    None,
    Basic,
    Balanced,
    Aggressive,
}

/// Information extracted from query analysis
#[derive(Debug, Clone)]
pub struct QueryInfo {
    pub query_type: QueryType,
    pub original_query: String,
    pub patterns: Vec<TriplePattern>,
    pub service_clauses: Vec<ServiceClause>,
    pub filters: Vec<FilterExpression>,
    pub variables: HashSet<String>,
    pub complexity: QueryComplexity,
    pub estimated_cost: u64,
}

/// Types of queries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryType {
    SparqlSelect,
    SparqlConstruct,
    SparqlAsk,
    SparqlDescribe,
    SparqlUpdate,
    GraphQLQuery,
    GraphQLMutation,
    GraphQLSubscription,
    Unknown,
}

/// RDF triple pattern
#[derive(Debug, Clone)]
pub struct TriplePattern {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub pattern_string: String,
}

/// SPARQL SERVICE clause
#[derive(Debug, Clone)]
pub struct ServiceClause {
    pub service_url: String,
    pub subquery: String,
    pub silent: bool,
}

/// SPARQL FILTER expression
#[derive(Debug, Clone)]
pub struct FilterExpression {
    pub expression: String,
    pub variables: HashSet<String>,
}

/// GraphQL selection
#[derive(Debug, Clone)]
pub struct GraphQLSelection {
    pub name: String,
    pub arguments: HashMap<String, serde_json::Value>,
    pub selections: Vec<GraphQLSelection>,
}

/// Query complexity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryComplexity {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Execution plan for federated queries
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub query_id: String,
    pub query_type: QueryType,
    pub steps: Vec<ExecutionStep>,
    pub estimated_duration: Duration,
    pub parallelizable_steps: Vec<Vec<String>>,
    pub dependencies: HashMap<String, Vec<String>>,
}

/// Individual step in an execution plan
#[derive(Debug, Clone)]
pub struct ExecutionStep {
    pub step_id: String,
    pub step_type: StepType,
    pub service_id: Option<String>,
    pub query_fragment: String,
    pub expected_variables: HashSet<String>,
    pub estimated_duration: Duration,
    pub dependencies: Vec<String>,
    pub parallel_group: Option<usize>,
}

/// Types of execution steps
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepType {
    ServiceQuery,
    GraphQLQuery,
    Join,
    Union,
    Filter,
    SchemaStitch,
    Aggregate,
    Sort,
}

impl std::fmt::Display for StepType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StepType::ServiceQuery => write!(f, "ServiceQuery"),
            StepType::GraphQLQuery => write!(f, "GraphQLQuery"),
            StepType::Join => write!(f, "Join"),
            StepType::Union => write!(f, "Union"),
            StepType::Filter => write!(f, "Filter"),
            StepType::SchemaStitch => write!(f, "SchemaStitch"),
            StepType::Aggregate => write!(f, "Aggregate"),
            StepType::Sort => write!(f, "Sort"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FederatedService, ServiceRegistry};

    #[tokio::test]
    async fn test_query_planner_creation() {
        let planner = QueryPlanner::new();
        assert_eq!(planner.config.max_services_per_query, 10);
    }

    #[tokio::test]
    async fn test_sparql_query_analysis() {
        let planner = QueryPlanner::new();
        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";

        let result = planner.analyze_sparql(query).await;
        assert!(result.is_ok());

        let query_info = result.unwrap();
        assert_eq!(query_info.query_type, QueryType::SparqlSelect);
        assert!(!query_info.variables.is_empty());
    }

    #[tokio::test]
    async fn test_query_type_detection() {
        let planner = QueryPlanner::new();

        assert_eq!(
            planner.detect_query_type("SELECT * WHERE { ?s ?p ?o }"),
            QueryType::SparqlSelect
        );
        assert_eq!(
            planner.detect_query_type("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"),
            QueryType::SparqlConstruct
        );
        assert_eq!(
            planner.detect_query_type("ASK { ?s ?p ?o }"),
            QueryType::SparqlAsk
        );
        assert_eq!(
            planner.detect_query_type("DESCRIBE <http://example.org>"),
            QueryType::SparqlDescribe
        );
    }

    #[tokio::test]
    async fn test_service_clause_extraction() {
        let planner = QueryPlanner::new();
        let query = "SELECT * WHERE { SERVICE <http://example.org/sparql> { ?s ?p ?o } }";

        let services = planner.extract_service_clauses(query).unwrap();
        assert_eq!(services.len(), 1);
        assert_eq!(services[0].service_url, "http://example.org/sparql");
    }

    #[tokio::test]
    async fn test_execution_plan_creation() {
        let planner = QueryPlanner::new();
        let mut registry = ServiceRegistry::new();

        let service = FederatedService::new_sparql(
            "test-service".to_string(),
            "Test Service".to_string(),
            "http://example.com/sparql".to_string(),
        );
        registry.register(service).await.unwrap();

        let query_info = QueryInfo {
            query_type: QueryType::SparqlSelect,
            original_query: "SELECT * WHERE { ?s ?p ?o }".to_string(),
            patterns: vec![TriplePattern {
                subject: "?s".to_string(),
                predicate: "?p".to_string(),
                object: "?o".to_string(),
                pattern_string: "?s ?p ?o".to_string(),
            }],
            service_clauses: Vec::new(),
            filters: Vec::new(),
            variables: ["?s", "?p", "?o"]
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            complexity: QueryComplexity::Low,
            estimated_cost: 10,
        };

        let plan = planner.plan_sparql(&query_info, &registry).await.unwrap();
        assert!(!plan.steps.is_empty());
        assert_eq!(plan.query_type, QueryType::SparqlSelect);
    }
}
