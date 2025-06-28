//! SERVICE Clause Optimization and Rewriting
//!
//! This module implements advanced optimization techniques for SPARQL SERVICE clauses,
//! including query pushdown, filter propagation, and intelligent service selection.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::{
    planner::{
        ExecutionPlan, FilterExpression, QueryInfo as PlannerQueryInfo,
        ServiceClause as PlannerServiceClause, TriplePattern,
    },
    query_decomposition::{DecompositionResult, QueryDecomposer},
    FederatedService, ServiceCapability, ServiceRegistry,
};

/// SERVICE clause optimizer
#[derive(Debug)]
pub struct ServiceOptimizer {
    config: ServiceOptimizerConfig,
    statistics_cache: Arc<StatisticsCache>,
}

impl ServiceOptimizer {
    /// Create a new SERVICE optimizer
    pub fn new() -> Self {
        Self {
            config: ServiceOptimizerConfig::default(),
            statistics_cache: Arc::new(StatisticsCache::new()),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ServiceOptimizerConfig) -> Self {
        Self {
            config,
            statistics_cache: Arc::new(StatisticsCache::new()),
        }
    }

    /// Optimize a query containing SERVICE clauses
    pub async fn optimize_query(
        &self,
        query_info: &PlannerQueryInfo,
        registry: &ServiceRegistry,
    ) -> Result<OptimizedQuery> {
        info!(
            "Optimizing query with {} SERVICE clauses",
            query_info.service_clauses.len()
        );

        let mut optimized_services = Vec::new();
        let mut global_filters = query_info.filters.clone();
        let mut cross_service_joins = Vec::new();

        // Convert planner SERVICE clauses to optimizer format
        let service_clauses: Vec<ServiceClause> = query_info
            .service_clauses
            .iter()
            .map(|sc| ServiceClause {
                endpoint: Some(sc.service_url.clone()),
                patterns: self.extract_patterns_from_subquery(&sc.subquery),
                filters: self.extract_filters_from_subquery(&sc.subquery),
                silent: sc.silent,
            })
            .collect();

        // Analyze SERVICE clauses
        for service_clause in &service_clauses {
            let optimized = self
                .optimize_service_clause(service_clause, &mut global_filters, registry)
                .await?;
            optimized_services.push(optimized);
        }

        // Identify cross-service joins
        cross_service_joins = self.identify_cross_service_joins(&optimized_services);

        // Apply global optimizations
        let execution_strategy =
            self.determine_execution_strategy(&optimized_services, &cross_service_joins);

        Ok(OptimizedQuery {
            services: optimized_services,
            global_filters,
            cross_service_joins,
            execution_strategy,
            estimated_cost: self.estimate_total_cost(&optimized_services),
        })
    }

    /// Optimize a single SERVICE clause
    async fn optimize_service_clause(
        &self,
        service_clause: &ServiceClause,
        global_filters: &mut Vec<FilterExpression>,
        registry: &ServiceRegistry,
    ) -> Result<OptimizedServiceClause> {
        debug!(
            "Optimizing SERVICE clause for: {:?}",
            service_clause.endpoint
        );

        // Resolve service endpoint
        let service = if let Some(endpoint) = &service_clause.endpoint {
            self.resolve_service(endpoint, registry).await?
        } else {
            return Err(anyhow!("SERVICE clause requires an endpoint"));
        };

        // Extract patterns and filters from the SERVICE clause
        let mut patterns = service_clause.patterns.clone();
        let mut local_filters = service_clause.filters.clone();

        // Apply filter pushdown
        let pushed_filters =
            self.push_down_filters(&patterns, global_filters, &mut local_filters, &service);

        // Optimize patterns for the service
        let optimized_patterns = self.optimize_patterns_for_service(&patterns, &service);

        // Determine execution strategy
        let strategy =
            self.determine_service_strategy(&service, &optimized_patterns, &local_filters);

        // Estimate cost
        let estimated_cost = self
            .estimate_service_cost(&service, &optimized_patterns, &local_filters)
            .await;

        Ok(OptimizedServiceClause {
            service_id: service.id.clone(),
            endpoint: service.endpoint.clone(),
            patterns: optimized_patterns,
            filters: local_filters,
            pushed_filters,
            strategy,
            estimated_cost,
            capabilities: service.capabilities.clone(),
        })
    }

    /// Resolve service from endpoint or identifier
    async fn resolve_service(
        &self,
        endpoint: &str,
        registry: &ServiceRegistry,
    ) -> Result<FederatedService> {
        // First check if it's a service ID
        if let Some(service) = registry.get_service(endpoint) {
            return Ok(service.clone());
        }

        // Check if it's a known endpoint URL
        for service in registry.get_all_services() {
            if service.endpoint == endpoint {
                return Ok(service.clone());
            }
        }

        // Try to discover service at endpoint
        let discovery = crate::ServiceDiscovery::new();
        if let Some(service) = discovery.discover_service_at_endpoint(endpoint).await? {
            return Ok(service);
        }

        Err(anyhow!("Service not found for endpoint: {}", endpoint))
    }

    /// Push down filters to SERVICE clauses
    fn push_down_filters(
        &self,
        patterns: &[TriplePattern],
        global_filters: &mut Vec<FilterExpression>,
        local_filters: &mut Vec<FilterExpression>,
        service: &FederatedService,
    ) -> Vec<FilterExpression> {
        let mut pushed_filters = Vec::new();

        // Extract variables used in patterns
        let pattern_vars: HashSet<String> = patterns
            .iter()
            .flat_map(|p| {
                let mut vars = Vec::new();
                if p.subject.starts_with('?') {
                    vars.push(p.subject.clone());
                }
                if p.predicate.starts_with('?') {
                    vars.push(p.predicate.clone());
                }
                if p.object.starts_with('?') {
                    vars.push(p.object.clone());
                }
                vars
            })
            .collect();

        // Check each global filter
        let mut remaining_global = Vec::new();
        for filter in global_filters.drain(..) {
            // Check if all variables in filter are bound by patterns
            let filter_vars_bound = filter
                .variables
                .iter()
                .all(|var| pattern_vars.contains(var));

            if filter_vars_bound && self.can_service_handle_filter(&filter, service) {
                // Push filter down to service
                local_filters.push(filter.clone());
                pushed_filters.push(filter);
            } else {
                // Keep filter at global level
                remaining_global.push(filter);
            }
        }

        *global_filters = remaining_global;
        pushed_filters
    }

    /// Check if service can handle a specific filter
    fn can_service_handle_filter(
        &self,
        filter: &FilterExpression,
        service: &FederatedService,
    ) -> bool {
        // Check for specific filter capabilities
        if filter.expression.contains("REGEX") {
            return service
                .capabilities
                .contains(&ServiceCapability::FullTextSearch);
        }

        if filter.expression.contains("geof:") || filter.expression.contains("DISTANCE") {
            return service
                .capabilities
                .contains(&ServiceCapability::Geospatial);
        }

        if filter.expression.contains("NOW()") || filter.expression.contains("YEAR(") {
            return service
                .capabilities
                .contains(&ServiceCapability::TemporalQueries);
        }

        // Basic filters can be handled by any SPARQL service
        service
            .capabilities
            .contains(&ServiceCapability::SparqlQuery)
    }

    /// Optimize patterns for a specific service
    fn optimize_patterns_for_service(
        &self,
        patterns: &[TriplePattern],
        service: &FederatedService,
    ) -> Vec<TriplePattern> {
        let mut optimized = patterns.to_vec();

        // Sort patterns by selectivity (most selective first)
        optimized.sort_by_key(|p| self.estimate_pattern_selectivity(p, service));

        // Group related patterns
        if self.config.enable_pattern_grouping {
            optimized = self.group_related_patterns(optimized);
        }

        // Apply service-specific optimizations
        if service.name.contains("wikidata") {
            optimized = self.optimize_for_wikidata(optimized);
        } else if service.name.contains("dbpedia") {
            optimized = self.optimize_for_dbpedia(optimized);
        }

        optimized
    }

    /// Estimate selectivity of a pattern for ordering
    fn estimate_pattern_selectivity(
        &self,
        pattern: &TriplePattern,
        service: &FederatedService,
    ) -> u32 {
        let mut score: u32 = 0;

        // Constants are more selective than variables
        if !pattern.subject.starts_with('?') {
            score += 100;
        }
        if !pattern.predicate.starts_with('?') {
            score += 200; // Predicates are usually most selective
        }
        if !pattern.object.starts_with('?') {
            score += 100;
        }

        // Use cached statistics if available
        if let Some(stats) = self
            .statistics_cache
            .get_predicate_stats(&pattern.predicate)
        {
            score = score.saturating_sub(stats.frequency / 1000);
        }

        score
    }

    /// Group related patterns for better execution
    fn group_related_patterns(&self, patterns: Vec<TriplePattern>) -> Vec<TriplePattern> {
        // Group patterns that share variables
        let mut grouped = Vec::new();
        let mut remaining = patterns;

        while !remaining.is_empty() {
            let seed = remaining.remove(0);
            grouped.push(seed.clone());

            let seed_vars = self.extract_pattern_variables(&seed);

            // Find patterns that share variables with seed
            let mut i = 0;
            while i < remaining.len() {
                let pattern_vars = self.extract_pattern_variables(&remaining[i]);

                if !seed_vars.is_disjoint(&pattern_vars) {
                    grouped.push(remaining.remove(i));
                } else {
                    i += 1;
                }
            }
        }

        grouped
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

    /// Optimize patterns specifically for Wikidata
    fn optimize_for_wikidata(&self, patterns: Vec<TriplePattern>) -> Vec<TriplePattern> {
        let mut optimized = patterns;

        // Wikidata-specific optimizations
        // 1. Use HINT:Query optimizer hints
        // 2. Prefer wdt: properties over full statements
        // 3. Use SERVICE wikibase:label for labels

        optimized
    }

    /// Optimize patterns specifically for DBpedia
    fn optimize_for_dbpedia(&self, patterns: Vec<TriplePattern>) -> Vec<TriplePattern> {
        let mut optimized = patterns;

        // DBpedia-specific optimizations
        // 1. Use indexed properties first
        // 2. Avoid expensive property paths
        // 3. Leverage DBpedia's category system

        optimized
    }

    /// Determine execution strategy for a service
    fn determine_service_strategy(
        &self,
        service: &FederatedService,
        patterns: &[TriplePattern],
        filters: &[FilterExpression],
    ) -> ServiceExecutionStrategy {
        // Check if we should use VALUES clause for bindings
        let use_values_binding = service
            .capabilities
            .contains(&ServiceCapability::SparqlValues)
            && patterns.len() < self.config.max_patterns_for_values;

        // Check if we should stream results
        let stream_results =
            self.estimate_result_size(patterns, filters) > self.config.streaming_threshold;

        // Check if we can use subqueries
        let use_subqueries = service
            .capabilities
            .contains(&ServiceCapability::SparqlSubqueries)
            && patterns.len() > self.config.min_patterns_for_subquery;

        ServiceExecutionStrategy {
            use_values_binding,
            stream_results,
            use_subqueries,
            batch_size: self.config.default_batch_size,
            timeout_ms: self.config.service_timeout_ms,
        }
    }

    /// Estimate result size for patterns and filters
    fn estimate_result_size(
        &self,
        patterns: &[TriplePattern],
        filters: &[FilterExpression],
    ) -> usize {
        let base_size = 1000;
        let pattern_factor = patterns.len();
        let filter_reduction = filters.len() * 10;

        base_size * pattern_factor / (filter_reduction + 1)
    }

    /// Estimate cost for executing patterns on a service
    async fn estimate_service_cost(
        &self,
        service: &FederatedService,
        patterns: &[TriplePattern],
        filters: &[FilterExpression],
    ) -> f64 {
        let mut cost = 0.0;

        // Base cost from service performance
        if let Some(avg_time) = service.performance.average_response_time {
            cost += avg_time.as_millis() as f64;
        } else {
            cost += 1000.0; // Default 1 second
        }

        // Pattern complexity
        cost += patterns.len() as f64 * 10.0;

        // Filter complexity
        cost += filters.len() as f64 * 5.0;

        // Network latency estimate
        cost += 50.0;

        // Use cached statistics if available
        for pattern in patterns {
            if let Some(stats) = self
                .statistics_cache
                .get_predicate_stats(&pattern.predicate)
            {
                cost += (stats.frequency as f64).log10() * 10.0;
            }
        }

        cost
    }

    /// Extract patterns from a subquery string
    fn extract_patterns_from_subquery(&self, subquery: &str) -> Vec<TriplePattern> {
        let mut patterns = Vec::new();

        // Simple pattern extraction - in real implementation would use proper SPARQL parser
        let lines: Vec<&str> = subquery.split('.').collect();

        for line in lines {
            let line = line.trim();
            if line.is_empty() || line.starts_with("FILTER") {
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

        patterns
    }

    /// Extract filters from a subquery string
    fn extract_filters_from_subquery(&self, subquery: &str) -> Vec<FilterExpression> {
        let mut filters = Vec::new();

        let mut pos = 0;
        while let Some(filter_pos) = subquery[pos..].find("FILTER") {
            let actual_pos = pos + filter_pos;

            if let Some(start) = subquery[actual_pos..].find('(') {
                let mut paren_count = 0;
                let mut end_pos = None;

                for (i, ch) in subquery[actual_pos + start..].char_indices() {
                    if ch == '(' {
                        paren_count += 1;
                    } else if ch == ')' {
                        paren_count -= 1;
                        if paren_count == 0 {
                            end_pos = Some(actual_pos + start + i + 1);
                            break;
                        }
                    }
                }

                if let Some(end) = end_pos {
                    let expression = subquery[actual_pos + start + 1..end - 1].trim().to_string();
                    let variables = self.extract_filter_variables(&expression);

                    filters.push(FilterExpression {
                        expression,
                        variables,
                    });
                }
            }

            pos = actual_pos + 6;
        }

        filters
    }

    /// Extract variables from a filter expression
    fn extract_filter_variables(&self, expression: &str) -> HashSet<String> {
        let mut variables = HashSet::new();

        for word in expression.split_whitespace() {
            if word.starts_with('?') {
                let clean_var = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '?');
                variables.insert(clean_var.to_string());
            }
        }

        variables
    }

    /// Identify joins between SERVICE clauses
    fn identify_cross_service_joins(
        &self,
        services: &[OptimizedServiceClause],
    ) -> Vec<CrossServiceJoin> {
        let mut joins = Vec::new();

        for i in 0..services.len() {
            for j in (i + 1)..services.len() {
                let service1 = &services[i];
                let service2 = &services[j];

                // Find shared variables
                let vars1 = self.extract_service_variables(service1);
                let vars2 = self.extract_service_variables(service2);

                let shared_vars: Vec<String> = vars1.intersection(&vars2).cloned().collect();

                if !shared_vars.is_empty() {
                    joins.push(CrossServiceJoin {
                        left_service: service1.service_id.clone(),
                        right_service: service2.service_id.clone(),
                        join_variables: shared_vars,
                        join_type: JoinType::Inner,
                        estimated_selectivity: 0.1, // Default estimate
                    });
                }
            }
        }

        joins
    }

    /// Extract all variables from a service clause
    fn extract_service_variables(&self, service: &OptimizedServiceClause) -> HashSet<String> {
        let mut vars = HashSet::new();

        for pattern in &service.patterns {
            vars.extend(self.extract_pattern_variables(pattern));
        }

        for filter in &service.filters {
            vars.extend(filter.variables.iter().cloned());
        }

        vars
    }

    /// Determine overall execution strategy
    fn determine_execution_strategy(
        &self,
        services: &[OptimizedServiceClause],
        joins: &[CrossServiceJoin],
    ) -> ExecutionStrategy {
        // Single service - simple execution
        if services.len() == 1 {
            return ExecutionStrategy::Sequential;
        }

        // No joins - parallel execution
        if joins.is_empty() {
            return ExecutionStrategy::Parallel;
        }

        // Complex joins - use adaptive strategy
        if joins.len() > 2 || self.has_complex_join_pattern(joins) {
            return ExecutionStrategy::Adaptive;
        }

        // Default to parallel with join
        ExecutionStrategy::ParallelWithJoin
    }

    /// Check if join pattern is complex
    fn has_complex_join_pattern(&self, joins: &[CrossServiceJoin]) -> bool {
        // Check for multi-way joins or low selectivity
        joins
            .iter()
            .any(|j| j.join_variables.len() > 2 || j.estimated_selectivity > 0.5)
    }

    /// Estimate total cost of execution
    fn estimate_total_cost(&self, services: &[OptimizedServiceClause]) -> f64 {
        services.iter().map(|s| s.estimated_cost).sum()
    }

    /// Apply SERVICE clause merging optimization
    pub fn merge_service_clauses(&self, services: &mut Vec<OptimizedServiceClause>) -> Result<()> {
        // Group services by endpoint
        let mut endpoint_groups: HashMap<String, Vec<usize>> = HashMap::new();

        for (idx, service) in services.iter().enumerate() {
            endpoint_groups
                .entry(service.endpoint.clone())
                .or_default()
                .push(idx);
        }

        // Merge services with same endpoint
        for (endpoint, indices) in endpoint_groups {
            if indices.len() > 1 && self.config.enable_service_merging {
                debug!(
                    "Merging {} SERVICE clauses for endpoint: {}",
                    indices.len(),
                    endpoint
                );

                // Merge patterns and filters
                let mut merged_patterns = Vec::new();
                let mut merged_filters = Vec::new();

                for &idx in &indices[1..] {
                    if let Some(service) = services.get(idx) {
                        merged_patterns.extend(service.patterns.clone());
                        merged_filters.extend(service.filters.clone());
                    }
                }

                // Update first service with merged data
                if let Some(first_service) = services.get_mut(indices[0]) {
                    first_service.patterns.extend(merged_patterns);
                    first_service.filters.extend(merged_filters);

                    // Deduplicate
                    first_service
                        .patterns
                        .sort_by(|a, b| a.pattern_string.cmp(&b.pattern_string));
                    first_service
                        .patterns
                        .dedup_by(|a, b| a.pattern_string == b.pattern_string);
                }
            }
        }

        Ok(())
    }
}

impl Default for ServiceOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for SERVICE optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceOptimizerConfig {
    /// Enable pattern grouping
    pub enable_pattern_grouping: bool,

    /// Enable SERVICE clause merging
    pub enable_service_merging: bool,

    /// Maximum patterns for VALUES binding
    pub max_patterns_for_values: usize,

    /// Minimum patterns for subquery
    pub min_patterns_for_subquery: usize,

    /// Result size threshold for streaming
    pub streaming_threshold: usize,

    /// Default batch size
    pub default_batch_size: usize,

    /// Service timeout in milliseconds
    pub service_timeout_ms: u64,

    /// Enable statistics-based optimization
    pub enable_statistics: bool,
}

impl Default for ServiceOptimizerConfig {
    fn default() -> Self {
        Self {
            enable_pattern_grouping: true,
            enable_service_merging: true,
            max_patterns_for_values: 10,
            min_patterns_for_subquery: 5,
            streaming_threshold: 10000,
            default_batch_size: 1000,
            service_timeout_ms: 30000,
            enable_statistics: true,
        }
    }
}

/// SERVICE clause representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceClause {
    /// Service endpoint or identifier
    pub endpoint: Option<String>,

    /// Patterns in the SERVICE clause
    pub patterns: Vec<TriplePattern>,

    /// Filters in the SERVICE clause
    pub filters: Vec<FilterExpression>,

    /// Whether SERVICE is SILENT
    pub silent: bool,
}

/// Optimized SERVICE clause
#[derive(Debug, Clone)]
pub struct OptimizedServiceClause {
    /// Resolved service ID
    pub service_id: String,

    /// Service endpoint
    pub endpoint: String,

    /// Optimized patterns
    pub patterns: Vec<TriplePattern>,

    /// Local filters
    pub filters: Vec<FilterExpression>,

    /// Filters pushed down from global scope
    pub pushed_filters: Vec<FilterExpression>,

    /// Execution strategy
    pub strategy: ServiceExecutionStrategy,

    /// Estimated execution cost
    pub estimated_cost: f64,

    /// Service capabilities
    pub capabilities: HashSet<ServiceCapability>,
}

/// Execution strategy for a SERVICE clause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceExecutionStrategy {
    /// Use VALUES clause for bindings
    pub use_values_binding: bool,

    /// Stream results
    pub stream_results: bool,

    /// Use subqueries
    pub use_subqueries: bool,

    /// Batch size for processing
    pub batch_size: usize,

    /// Timeout in milliseconds
    pub timeout_ms: u64,
}

/// Cross-service join information
#[derive(Debug, Clone)]
pub struct CrossServiceJoin {
    /// Left service ID
    pub left_service: String,

    /// Right service ID
    pub right_service: String,

    /// Join variables
    pub join_variables: Vec<String>,

    /// Join type
    pub join_type: JoinType,

    /// Estimated selectivity
    pub estimated_selectivity: f64,
}

/// Join types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    LeftOuter,
    RightOuter,
    Full,
}

/// Overall execution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStrategy {
    Sequential,
    Parallel,
    ParallelWithJoin,
    Adaptive,
}

/// Optimized query result
#[derive(Debug)]
pub struct OptimizedQuery {
    /// Optimized SERVICE clauses
    pub services: Vec<OptimizedServiceClause>,

    /// Remaining global filters
    pub global_filters: Vec<FilterExpression>,

    /// Cross-service joins
    pub cross_service_joins: Vec<CrossServiceJoin>,

    /// Overall execution strategy
    pub execution_strategy: ExecutionStrategy,

    /// Total estimated cost
    pub estimated_cost: f64,
}

/// Statistics cache for optimization decisions
#[derive(Debug)]
struct StatisticsCache {
    predicate_stats: parking_lot::RwLock<HashMap<String, PredicateStatistics>>,
}

impl StatisticsCache {
    fn new() -> Self {
        Self {
            predicate_stats: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    fn get_predicate_stats(&self, predicate: &str) -> Option<PredicateStatistics> {
        self.predicate_stats.read().get(predicate).cloned()
    }

    fn update_predicate_stats(&self, predicate: String, stats: PredicateStatistics) {
        self.predicate_stats.write().insert(predicate, stats);
    }
}

/// Statistics for a predicate
#[derive(Debug, Clone)]
struct PredicateStatistics {
    /// Frequency of the predicate
    pub frequency: u64,

    /// Average number of objects per subject
    pub avg_objects_per_subject: f64,

    /// Selectivity estimate
    pub selectivity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_optimizer_creation() {
        let optimizer = ServiceOptimizer::new();
        assert!(optimizer.config.enable_pattern_grouping);
    }

    #[test]
    fn test_pattern_variable_extraction() {
        let optimizer = ServiceOptimizer::new();
        let pattern = TriplePattern {
            subject: "?s".to_string(),
            predicate: "rdf:type".to_string(),
            object: "?o".to_string(),
            pattern_string: "?s rdf:type ?o".to_string(),
        };

        let vars = optimizer.extract_pattern_variables(&pattern);
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("?s"));
        assert!(vars.contains("?o"));
    }

    #[test]
    fn test_execution_strategy_determination() {
        let optimizer = ServiceOptimizer::new();

        // Single service
        let services = vec![OptimizedServiceClause {
            service_id: "test".to_string(),
            endpoint: "http://example.com/sparql".to_string(),
            patterns: vec![],
            filters: vec![],
            pushed_filters: vec![],
            strategy: ServiceExecutionStrategy {
                use_values_binding: false,
                stream_results: false,
                use_subqueries: false,
                batch_size: 1000,
                timeout_ms: 30000,
            },
            estimated_cost: 100.0,
            capabilities: HashSet::new(),
        }];

        let joins = vec![];
        let strategy = optimizer.determine_execution_strategy(&services, &joins);
        assert_eq!(strategy, ExecutionStrategy::Sequential);
    }
}
