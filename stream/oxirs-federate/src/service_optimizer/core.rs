//! Core SERVICE optimization implementation
//!
//! This module contains the main ServiceOptimizer implementation with core optimization methods.

use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::{debug, info};

use crate::{
    planner::{FilterExpression, QueryInfo, TriplePattern},
    service_registry::ServiceRegistry,
    FederatedService, ServiceCapability,
};

use super::types::*;

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
        query_info: &QueryInfo,
        registry: &ServiceRegistry,
    ) -> Result<OptimizedQuery> {
        // Extract SERVICE clauses from the original query (simplified approach)
        let service_clauses = self.extract_service_clauses_from_query(&query_info.original_query);

        info!(
            "Optimizing query with {} SERVICE clauses",
            service_clauses.len()
        );

        let mut optimized_services = Vec::new();
        let mut global_filters = query_info.filters.clone();

        // Analyze SERVICE clauses
        for service_clause in &service_clauses {
            let optimized = self
                .optimize_service_clause(service_clause, &mut global_filters, registry)
                .await?;
            optimized_services.push(optimized);
        }

        // Identify cross-service joins
        let cross_service_joins = self.identify_cross_service_joins(&optimized_services);

        // Apply global optimizations
        let execution_strategy =
            self.determine_execution_strategy(&optimized_services, &cross_service_joins);

        let estimated_cost = self.estimate_total_cost(&optimized_services);
        Ok(OptimizedQuery {
            services: optimized_services.clone(),
            global_filters,
            cross_service_joins,
            execution_strategy,
            estimated_cost,
        })
    }

    /// Optimize a single SERVICE clause
    pub async fn optimize_service_clause(
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
        let patterns = service_clause.patterns.clone();
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
        let discovery = crate::discovery::ServiceDiscovery::new();
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
                if let Some(ref subject) = p.subject {
                    if subject.starts_with('?') {
                        vars.push(subject.clone());
                    }
                }
                if let Some(ref predicate) = p.predicate {
                    if predicate.starts_with('?') {
                        vars.push(predicate.clone());
                    }
                }
                if let Some(ref object) = p.object {
                    if object.starts_with('?') {
                        vars.push(object.clone());
                    }
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
        _service: &FederatedService,
    ) -> u32 {
        let mut score: u32 = 0;

        // Constants are more selective than variables
        if let Some(subject) = &pattern.subject {
            if !subject.starts_with('?') {
                score += 100;
            }
        }
        if let Some(predicate) = &pattern.predicate {
            if !predicate.starts_with('?') {
                score += 200; // Predicates are usually most selective
            }
        }
        if let Some(object) = &pattern.object {
            if !object.starts_with('?') {
                score += 100;
            }
        }

        // Use cached statistics if available
        if let Some(predicate) = &pattern.predicate {
            if let Some(stats) = self.statistics_cache.get_predicate_stats(predicate) {
                score = score.saturating_sub((stats.frequency / 1000) as u32);
            }
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
    pub fn extract_pattern_variables(&self, pattern: &TriplePattern) -> HashSet<String> {
        let mut vars = HashSet::new();

        if let Some(subject) = &pattern.subject {
            if subject.starts_with('?') {
                vars.insert(subject.clone());
            }
        }
        if let Some(predicate) = &pattern.predicate {
            if predicate.starts_with('?') {
                vars.insert(predicate.clone());
            }
        }
        if let Some(object) = &pattern.object {
            if object.starts_with('?') {
                vars.insert(object.clone());
            }
        }

        vars
    }

    /// Optimize patterns specifically for Wikidata
    fn optimize_for_wikidata(&self, patterns: Vec<TriplePattern>) -> Vec<TriplePattern> {
        // Wikidata-specific optimizations
        // 1. Use HINT:Query optimizer hints
        // 2. Prefer wdt: properties over full statements
        // 3. Use SERVICE wikibase:label for labels

        patterns
    }

    /// Optimize patterns specifically for DBpedia
    fn optimize_for_dbpedia(&self, patterns: Vec<TriplePattern>) -> Vec<TriplePattern> {
        // DBpedia-specific optimizations
        // 1. Use indexed properties first
        // 2. Avoid expensive property paths
        // 3. Leverage DBpedia's category system

        patterns
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
    pub async fn estimate_service_cost(
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
            if let Some(predicate) = &pattern.predicate {
                if let Some(stats) = self.statistics_cache.get_predicate_stats(predicate) {
                    cost += (stats.frequency as f64).log10() * 10.0;
                }
            }
        }

        cost
    }

    /// Extract patterns from a subquery string
    #[allow(dead_code)]
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
                    subject: Some(parts[0].to_string()),
                    predicate: Some(parts[1].to_string()),
                    object: Some(parts[2..].join(" ")),
                    pattern_string: line.to_string(),
                });
            }
        }

        patterns
    }

    /// Extract filters from a subquery string
    #[allow(dead_code)]
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
                        variables: variables.into_iter().collect(),
                    });
                }
            }

            pos = actual_pos + 6;
        }

        filters
    }

    /// Extract variables from a filter expression
    #[allow(dead_code)]
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

    /// Determine overall execution strategy
    pub fn determine_execution_strategy(
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
    pub fn merge_service_clauses(&self, services: &mut [OptimizedServiceClause]) -> Result<()> {
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

    /// Extract SERVICE clauses from a SPARQL query (simplified implementation)
    pub fn extract_service_clauses_from_query(&self, query: &str) -> Vec<ServiceClause> {
        let mut service_clauses = Vec::new();

        // Simple pattern matching to find SERVICE clauses
        // In a production implementation, this would use a proper SPARQL parser
        for line in query.lines() {
            let line = line.trim();
            if line.to_uppercase().starts_with("SERVICE") {
                // Extract endpoint from SERVICE <endpoint> { ... }
                if let Some(start) = line.find('<') {
                    if let Some(end) = line.find('>') {
                        let endpoint = line[start + 1..end].to_string();
                        service_clauses.push(ServiceClause {
                            endpoint: Some(endpoint),
                            patterns: vec![], // Simplified - would extract patterns from the SERVICE block
                            filters: vec![], // Simplified - would extract filters from the SERVICE block
                            silent: line.to_uppercase().contains("SILENT"),
                        });
                    }
                }
            }
        }

        // If no SERVICE clauses found, create a default one for compatibility
        if service_clauses.is_empty() {
            service_clauses.push(ServiceClause {
                endpoint: None,
                patterns: vec![],
                filters: vec![],
                silent: false,
            });
        }

        service_clauses
    }

    /// Check if services are independent (no shared variables)
    #[allow(dead_code)]
    fn are_services_independent(&self, services: &[OptimizedServiceClause]) -> bool {
        for i in 0..services.len() {
            for j in i + 1..services.len() {
                let vars_i = self.extract_service_variables(&services[i]);
                let vars_j = self.extract_service_variables(&services[j]);

                // If services share variables, they're not independent
                if !vars_i.is_disjoint(&vars_j) {
                    return false;
                }
            }
        }
        true
    }

    /// Get predicate statistics from cache
    pub fn get_predicate_stats(&self, predicate: &str) -> Option<PredicateStatistics> {
        self.statistics_cache.get_predicate_stats(predicate)
    }

    /// Update service performance metrics
    pub fn update_service_performance(
        &self,
        service_id: &str,
        _metrics: &ServicePerformanceMetrics,
    ) {
        // Note: Arc<StatisticsCache> doesn't allow mutable access, this is a design issue
        // For now, we'll need to make StatisticsCache methods work with Arc/Mutex
        // self.statistics_cache.update_service_performance(service_id, metrics);
        debug!("Service performance update requested for {}", service_id);
    }

    /// Update service ranking
    pub fn update_service_ranking(&self, service_id: &str, ranking: f64) {
        // Note: Arc<StatisticsCache> doesn't allow mutable access, this is a design issue
        // For now, we'll need to make StatisticsCache methods work with Arc/Mutex
        // self.statistics_cache.update_service_ranking(service_id, ranking);
        debug!(
            "Service ranking update requested for {}: {}",
            service_id, ranking
        );
    }

    /// Estimate service cost for a single pattern
    pub fn estimate_single_pattern_service_cost(
        &self,
        pattern: &TriplePattern,
        _service: &FederatedService,
    ) -> f64 {
        // Basic cost estimation based on pattern complexity and service performance
        let base_cost = 1.0;
        let complexity_factor = match self.calculate_pattern_complexity(pattern) {
            PatternComplexity::Simple => 0.5,
            PatternComplexity::Medium => 1.0,
            PatternComplexity::Complex => 2.0,
        };
        base_cost * complexity_factor
    }

    /// Calculate pattern complexity for cost estimation
    pub fn calculate_pattern_complexity(&self, pattern: &TriplePattern) -> PatternComplexity {
        // Simple heuristic: more variables = more complex
        let var_count = [&pattern.subject, &pattern.predicate, &pattern.object]
            .iter()
            .filter(|term| term.as_ref().is_some_and(|t| t.starts_with('?')))
            .count();

        match var_count {
            0..=1 => PatternComplexity::Simple,
            2 => PatternComplexity::Medium,
            _ => PatternComplexity::Complex,
        }
    }

    /// ML-based result size estimation (stub implementation)
    pub fn estimate_result_size_ml(
        &self,
        pattern: &TriplePattern,
        service: &FederatedService,
        registry: &ServiceRegistry,
    ) -> Result<u64> {
        // Extract features for ML prediction
        let features = self.extract_ml_features(pattern, service, registry)?;

        // Use trained ML model for prediction
        let predicted_size = self.predict_with_ml_model(&features)?;

        // Apply bounds checking
        let bounded_size = predicted_size.clamp(1, 1_000_000); // Between 1 and 1M results

        debug!(
            "ML prediction for pattern {:?}: {} results (features: {:?})",
            pattern, bounded_size, features
        );

        Ok(bounded_size)
    }

    /// Extract machine learning features from pattern and service
    fn extract_ml_features(
        &self,
        pattern: &TriplePattern,
        service: &FederatedService,
        registry: &ServiceRegistry,
    ) -> Result<MLFeatures> {
        let mut features = MLFeatures::default();

        // Pattern complexity features
        let complexity = self.calculate_pattern_complexity(pattern);
        features.pattern_complexity = match complexity {
            PatternComplexity::Simple => 1.0,
            PatternComplexity::Medium => 2.0,
            PatternComplexity::Complex => 3.0,
        };

        // Variable count in pattern
        features.variable_count = self.count_variables_in_pattern(pattern) as f64;

        // Service-specific features
        features.service_load = self.get_ml_service_load_factor(service, registry) as f64;
        features.service_response_time = self.get_average_response_time(service, registry) as f64;

        // Dataset size estimation
        features.estimated_dataset_size =
            self.estimate_service_dataset_size(service, registry) as f64;

        // Pattern specificity (how specific vs. how general the pattern is)
        features.pattern_specificity = self.calculate_pattern_specificity(pattern);

        // Historical performance
        features.historical_avg_size = self.get_historical_average_size(pattern, service) as f64;

        Ok(features)
    }

    /// Predict result size using ML model (simple linear regression for now)
    fn predict_with_ml_model(&self, features: &MLFeatures) -> Result<u64> {
        // Simple linear regression model weights (would be learned from training data)
        let weights = MLModelWeights {
            intercept: 100.0,
            pattern_complexity: 500.0,
            variable_count: 200.0,
            service_load: -50.0, // More load = fewer results expected
            service_response_time: 0.1,
            estimated_dataset_size: 0.001,
            pattern_specificity: -300.0, // More specific = fewer results
            historical_avg_size: 0.7,    // Strong predictor
        };

        // Linear prediction: y = intercept + sum(weight_i * feature_i)
        let prediction = weights.intercept
            + weights.pattern_complexity * features.pattern_complexity
            + weights.variable_count * features.variable_count
            + weights.service_load * features.service_load
            + weights.service_response_time * features.service_response_time
            + weights.estimated_dataset_size * features.estimated_dataset_size
            + weights.pattern_specificity * features.pattern_specificity
            + weights.historical_avg_size * features.historical_avg_size;

        // Apply non-linear activation (sigmoid-like) to bound the result
        let bounded_prediction =
            (prediction.max(0.0) / (1.0 + (-prediction / 1000.0).exp())) * 10000.0;

        Ok(bounded_prediction as u64)
    }

    /// Count variables in a triple pattern
    fn count_variables_in_pattern(&self, pattern: &TriplePattern) -> usize {
        let mut count = 0;
        if let Some(subject) = &pattern.subject {
            if subject.starts_with('?') || subject.starts_with('$') {
                count += 1;
            }
        }
        if let Some(predicate) = &pattern.predicate {
            if predicate.starts_with('?') || predicate.starts_with('$') {
                count += 1;
            }
        }
        if let Some(object) = &pattern.object {
            if object.starts_with('?') || object.starts_with('$') {
                count += 1;
            }
        }
        count
    }

    /// Calculate pattern specificity (0.0 = very general, 1.0 = very specific)
    fn calculate_pattern_specificity(&self, pattern: &TriplePattern) -> f64 {
        let mut specificity = 0.0;

        // Concrete subject increases specificity
        if let Some(subject) = &pattern.subject {
            if !subject.starts_with('?') && !subject.starts_with('$') {
                specificity += 0.4;
            }
        }

        // Concrete predicate increases specificity
        if let Some(predicate) = &pattern.predicate {
            if !predicate.starts_with('?') && !predicate.starts_with('$') {
                specificity += 0.3;
            }
        }

        // Concrete object increases specificity
        if let Some(object) = &pattern.object {
            if !object.starts_with('?') && !object.starts_with('$') {
                specificity += 0.3;
            }
        }

        specificity
    }

    /// Get service load factor for ML features (0-100)
    fn get_ml_service_load_factor(
        &self,
        service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> u32 {
        // Simple heuristic based on recent query count
        // In real implementation, would query service metrics
        match service.capabilities.len() {
            0..=5 => 20,  // Low load
            6..=15 => 50, // Medium load
            _ => 80,      // High load
        }
    }

    /// Get average response time for service (in milliseconds)
    fn get_average_response_time(
        &self,
        _service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> u32 {
        // Would be tracked from actual service calls
        // For now, return a reasonable default
        150
    }

    /// Estimate total dataset size for service
    fn estimate_service_dataset_size(
        &self,
        service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> u64 {
        // Heuristic based on service capabilities and type
        use crate::service::ServiceType;
        let base_size = match service.service_type {
            ServiceType::Sparql => 10_000_000,   // Medium SPARQL endpoint
            ServiceType::GraphQL => 5_000_000,   // Medium GraphQL service
            ServiceType::Hybrid => 15_000_000,   // Larger hybrid service
            ServiceType::RestRdf => 1_000_000,   // Smaller REST API
            ServiceType::Custom(_) => 5_000_000, // Default for custom services
        };

        // Adjust based on capabilities
        let capability_factor = (service.capabilities.len() as f64 / 10.0).min(2.0);
        (base_size as f64 * capability_factor) as u64
    }

    /// Get historical average result size for similar patterns
    fn get_historical_average_size(
        &self,
        pattern: &TriplePattern,
        service: &FederatedService,
    ) -> u64 {
        // Look up in statistics cache
        let cache_key = format!("{}:{}", service.id, self.pattern_to_cache_key(pattern));

        self.statistics_cache
            .get_historical_size(&cache_key)
            .unwrap_or_else(|| {
                // Fallback: estimate based on pattern complexity
                match self.calculate_pattern_complexity(pattern) {
                    PatternComplexity::Simple => 5000,
                    PatternComplexity::Medium => 1000,
                    PatternComplexity::Complex => 200,
                }
            })
    }

    /// Convert pattern to cache key
    fn pattern_to_cache_key(&self, pattern: &TriplePattern) -> String {
        // Create a normalized pattern key for caching
        let subject_type = if let Some(subject) = &pattern.subject {
            if subject.starts_with('?') || subject.starts_with('$') {
                "VAR"
            } else {
                "CONST"
            }
        } else {
            "NONE"
        };

        let predicate_type = if let Some(predicate) = &pattern.predicate {
            if predicate.starts_with('?') || predicate.starts_with('$') {
                "VAR"
            } else {
                "CONST"
            }
        } else {
            "NONE"
        };

        let object_type = if let Some(object) = &pattern.object {
            if object.starts_with('?') || object.starts_with('$') {
                "VAR"
            } else {
                "CONST"
            }
        } else {
            "NONE"
        };

        format!("{subject_type}:{predicate_type}:{object_type}")
    }

    /// Estimate range selectivity factor for numeric/temporal predicates
    pub fn estimate_range_selectivity_factor(
        &self,
        _pattern: &TriplePattern,
        _service: &FederatedService,
    ) -> Result<f64> {
        // TODO: Implement range-based selectivity analysis
        // For now, return a default factor
        Ok(1.0)
    }

    /// Extract all variables used by a service
    pub fn extract_service_variables(&self, service: &OptimizedServiceClause) -> HashSet<String> {
        let mut variables = HashSet::new();
        for pattern in &service.patterns {
            variables.extend(self.extract_pattern_variables(pattern));
        }
        variables
    }

    /// Get the configuration
    pub fn config(&self) -> &ServiceOptimizerConfig {
        &self.config
    }
}

impl Default for ServiceOptimizer {
    fn default() -> Self {
        Self::new()
    }
}
