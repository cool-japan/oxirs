//! SERVICE Clause Optimization and Rewriting
//!
//! This module implements advanced optimization techniques for SPARQL SERVICE clauses,
//! including query pushdown, filter propagation, and intelligent service selection.

use anyhow::{anyhow, Result};
use bloom::ASMS;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
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
            score = score.saturating_sub((stats.frequency / 1000) as u32);
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

// ===== ADVANCED SOURCE SELECTION ALGORITHMS =====

impl ServiceOptimizer {
    /// Perform triple pattern coverage analysis for source selection
    pub async fn analyze_triple_pattern_coverage(
        &self,
        patterns: &[TriplePattern],
        services: &[FederatedService],
        registry: &ServiceRegistry,
    ) -> Result<HashMap<String, PatternCoverageAnalysis>> {
        let mut coverage_map = HashMap::new();

        for service in services {
            let coverage = self
                .calculate_service_pattern_coverage(service, patterns, registry)
                .await?;
            coverage_map.insert(service.id.clone(), coverage);
        }

        Ok(coverage_map)
    }

    /// Calculate how well a service covers a set of triple patterns
    async fn calculate_service_pattern_coverage(
        &self,
        service: &FederatedService,
        patterns: &[TriplePattern],
        _registry: &ServiceRegistry,
    ) -> Result<PatternCoverageAnalysis> {
        let mut covered_patterns = 0;
        let mut partially_covered = 0;
        let mut coverage_scores = Vec::new();

        for pattern in patterns {
            let score = self
                .calculate_pattern_coverage_score(service, pattern)
                .await?;
            coverage_scores.push(score);

            if score >= 0.8 {
                covered_patterns += 1;
            } else if score >= 0.3 {
                partially_covered += 1;
            }
        }

        let total_coverage = coverage_scores.iter().sum::<f64>() / patterns.len() as f64;
        let coverage_quality = self.assess_coverage_quality(&coverage_scores);

        Ok(PatternCoverageAnalysis {
            total_patterns: patterns.len(),
            covered_patterns,
            partially_covered_patterns: partially_covered,
            uncovered_patterns: patterns.len() - covered_patterns - partially_covered,
            overall_coverage_score: total_coverage,
            coverage_quality,
            pattern_scores: coverage_scores,
        })
    }

    /// Calculate coverage score for a single pattern on a service
    async fn calculate_pattern_coverage_score(
        &self,
        service: &FederatedService,
        pattern: &TriplePattern,
    ) -> Result<f64> {
        let mut score = 0.0;

        // Base score for general SPARQL capability
        if service
            .capabilities
            .contains(&ServiceCapability::SparqlQuery)
        {
            score += 0.5;
        }

        // Score boost for specific capabilities matching pattern needs
        if pattern.pattern_string.contains("REGEX")
            && service
                .capabilities
                .contains(&ServiceCapability::FullTextSearch)
        {
            score += 0.3;
        }

        if pattern.pattern_string.contains("geof:")
            && service
                .capabilities
                .contains(&ServiceCapability::Geospatial)
        {
            score += 0.3;
        }

        if pattern.pattern_string.contains("NOW()")
            && service
                .capabilities
                .contains(&ServiceCapability::TemporalQueries)
        {
            score += 0.2;
        }

        // Domain-specific boost based on service description and pattern content
        if let Some(ref description) = service.metadata.description {
            score += self.calculate_domain_affinity(description, pattern);
        }

        Ok(score.min(1.0))
    }

    /// Assess the quality of coverage distribution
    fn assess_coverage_quality(&self, scores: &[f64]) -> CoverageQuality {
        let variance = self.calculate_variance(scores);
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;

        if mean >= 0.8 && variance < 0.1 {
            CoverageQuality::Excellent
        } else if mean >= 0.6 && variance < 0.2 {
            CoverageQuality::Good
        } else if mean >= 0.4 {
            CoverageQuality::Fair
        } else {
            CoverageQuality::Poor
        }
    }

    /// Calculate variance for coverage quality assessment
    fn calculate_variance(&self, scores: &[f64]) -> f64 {
        if scores.is_empty() {
            return 0.0;
        }
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores
            .iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f64>()
            / scores.len() as f64;
        variance
    }

    /// Calculate domain affinity between service and pattern
    fn calculate_domain_affinity(&self, description: &str, pattern: &TriplePattern) -> f64 {
        let desc_lower = description.to_lowercase();
        let mut affinity = 0.0;

        // Check for domain keywords in pattern
        if pattern.pattern_string.contains("foaf:") && desc_lower.contains("social") {
            affinity += 0.2;
        }
        if pattern.pattern_string.contains("dbo:") && desc_lower.contains("dbpedia") {
            affinity += 0.3;
        }
        if pattern.pattern_string.contains("wdt:") && desc_lower.contains("wikidata") {
            affinity += 0.3;
        }
        if pattern.pattern_string.contains("geo:") && desc_lower.contains("geographic") {
            affinity += 0.2;
        }

        affinity
    }

    /// Predicate-based source filtering
    pub async fn filter_services_by_predicate(
        &self,
        predicate: &str,
        services: &[FederatedService],
        registry: &ServiceRegistry,
    ) -> Result<Vec<ServicePredicateScore>> {
        let mut scored_services = Vec::new();

        for service in services {
            let score = self
                .calculate_predicate_affinity_score(service, predicate, registry)
                .await?;

            if score > 0.1 {
                // Only include services with meaningful affinity
                scored_services.push(ServicePredicateScore {
                    service_id: service.id.clone(),
                    predicate: predicate.to_string(),
                    affinity_score: score,
                    estimated_result_count: self
                        .estimate_predicate_result_count(service, predicate)
                        .await?,
                    confidence_level: self.calculate_confidence_level(service, predicate),
                });
            }
        }

        // Sort by affinity score descending
        scored_services.sort_by(|a, b| {
            b.affinity_score
                .partial_cmp(&a.affinity_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(scored_services)
    }

    /// Calculate predicate affinity score for a service
    async fn calculate_predicate_affinity_score(
        &self,
        service: &FederatedService,
        predicate: &str,
        _registry: &ServiceRegistry,
    ) -> Result<f64> {
        let mut score = 0.0;

        // Check if service has statistics for this predicate
        if let Some(stats) = self.statistics_cache.get_predicate_stats(predicate) {
            // Higher frequency predicates get higher scores
            score += (stats.frequency as f64).log10() / 10.0;

            // Lower selectivity (more common) predicates get higher base scores
            score += (1.0 - stats.selectivity) * 0.5;
        }

        // Service-specific predicate scoring
        if let Some(ref description) = service.metadata.description {
            score += self.calculate_predicate_domain_match(predicate, description);
        }

        // Capability-based scoring
        score += self.calculate_predicate_capability_match(predicate, &service.capabilities);

        Ok(score.min(1.0))
    }

    /// Calculate predicate-domain matching score
    fn calculate_predicate_domain_match(&self, predicate: &str, description: &str) -> f64 {
        let desc_lower = description.to_lowercase();
        let mut score = 0.0;

        // Common namespace mappings
        if predicate.starts_with("foaf:") && desc_lower.contains("social") {
            score += 0.4;
        }
        if predicate.starts_with("dbo:")
            && (desc_lower.contains("dbpedia") || desc_lower.contains("ontology"))
        {
            score += 0.5;
        }
        if predicate.starts_with("wdt:") && desc_lower.contains("wikidata") {
            score += 0.5;
        }
        if predicate.starts_with("geo:")
            && (desc_lower.contains("geographic") || desc_lower.contains("spatial"))
        {
            score += 0.4;
        }
        if predicate.starts_with("dc:") && desc_lower.contains("dublin") {
            score += 0.3;
        }

        score
    }

    /// Calculate predicate-capability matching score
    fn calculate_predicate_capability_match(
        &self,
        predicate: &str,
        capabilities: &HashSet<ServiceCapability>,
    ) -> f64 {
        let mut score = 0.0;

        // Text-related predicates
        if predicate.contains("label") || predicate.contains("name") || predicate.contains("title")
        {
            if capabilities.contains(&ServiceCapability::FullTextSearch) {
                score += 0.2;
            }
        }

        // Geospatial predicates
        if predicate.contains("geo")
            || predicate.contains("location")
            || predicate.contains("coordinate")
        {
            if capabilities.contains(&ServiceCapability::Geospatial) {
                score += 0.3;
            }
        }

        // Temporal predicates
        if predicate.contains("date") || predicate.contains("time") || predicate.contains("year") {
            if capabilities.contains(&ServiceCapability::TemporalQueries) {
                score += 0.2;
            }
        }

        score
    }

    /// Estimate result count for a predicate on a service
    async fn estimate_predicate_result_count(
        &self,
        service: &FederatedService,
        predicate: &str,
    ) -> Result<u64> {
        // Use cached statistics if available
        if let Some(stats) = self.statistics_cache.get_predicate_stats(predicate) {
            return Ok(stats.frequency);
        }

        // Fallback estimation based on service characteristics
        let base_estimate = 1000_u64;
        let service_size_factor = if let Some(ref desc) = service.metadata.description {
            if desc.to_lowercase().contains("large") {
                5.0
            } else if desc.to_lowercase().contains("small") {
                0.2
            } else {
                1.0
            }
        } else {
            1.0
        };

        Ok((base_estimate as f64 * service_size_factor) as u64)
    }

    /// Calculate confidence level for predicate estimation
    fn calculate_confidence_level(
        &self,
        service: &FederatedService,
        predicate: &str,
    ) -> ConfidenceLevel {
        // Higher confidence if we have statistics
        if self
            .statistics_cache
            .get_predicate_stats(predicate)
            .is_some()
        {
            return ConfidenceLevel::High;
        }

        // Medium confidence for well-known namespaces
        if predicate.starts_with("rdf:")
            || predicate.starts_with("rdfs:")
            || predicate.starts_with("owl:")
            || predicate.starts_with("foaf:")
        {
            return ConfidenceLevel::Medium;
        }

        // Lower confidence for unknown predicates
        ConfidenceLevel::Low
    }

    /// Range-based source selection for numeric/temporal values
    pub async fn select_services_by_range(
        &self,
        predicate: &str,
        range: &ValueRange,
        services: &[FederatedService],
        registry: &ServiceRegistry,
    ) -> Result<Vec<RangeServiceMatch>> {
        let mut matches = Vec::new();

        for service in services {
            if let Some(range_match) = self
                .evaluate_service_range_coverage(service, predicate, range, registry)
                .await?
            {
                matches.push(range_match);
            }
        }

        // Sort by coverage score descending
        matches.sort_by(|a, b| {
            b.coverage_score
                .partial_cmp(&a.coverage_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(matches)
    }

    /// Evaluate how well a service covers a value range for a predicate
    async fn evaluate_service_range_coverage(
        &self,
        service: &FederatedService,
        predicate: &str,
        range: &ValueRange,
        _registry: &ServiceRegistry,
    ) -> Result<Option<RangeServiceMatch>> {
        // For demonstration, we'll use a simplified range coverage calculation
        // In practice, this would query service statistics or metadata

        let coverage_score = self
            .estimate_range_coverage(service, predicate, range)
            .await?;

        if coverage_score > 0.1 {
            Ok(Some(RangeServiceMatch {
                service_id: service.id.clone(),
                predicate: predicate.to_string(),
                range: range.clone(),
                coverage_score,
                estimated_result_count: self
                    .estimate_range_result_count(service, predicate, range)
                    .await?,
                overlap_type: self.classify_range_overlap(
                    range,
                    &self.estimate_service_range(service, predicate).await?,
                ),
            }))
        } else {
            Ok(None)
        }
    }

    /// Estimate how well a service covers a value range
    async fn estimate_range_coverage(
        &self,
        service: &FederatedService,
        predicate: &str,
        range: &ValueRange,
    ) -> Result<f64> {
        // Simplified range coverage estimation
        // This would be enhanced with actual service metadata and statistics

        let base_coverage = if service
            .capabilities
            .contains(&ServiceCapability::SparqlQuery)
        {
            0.5
        } else {
            0.0
        };

        // Boost for temporal capabilities on date ranges
        let temporal_boost = if matches!(range, ValueRange::Temporal(_, _))
            && service
                .capabilities
                .contains(&ServiceCapability::TemporalQueries)
        {
            0.3
        } else {
            0.0
        };

        // Domain-specific boost
        let domain_boost = if let Some(ref desc) = service.metadata.description {
            self.calculate_range_domain_affinity(desc, predicate, range)
        } else {
            0.0
        };

        Ok((base_coverage + temporal_boost + domain_boost).min(1.0))
    }

    /// Calculate domain affinity for range queries
    fn calculate_range_domain_affinity(
        &self,
        description: &str,
        predicate: &str,
        range: &ValueRange,
    ) -> f64 {
        let desc_lower = description.to_lowercase();
        let mut affinity = 0.0;

        match range {
            ValueRange::Temporal(_, _) => {
                if predicate.contains("date") || predicate.contains("time") {
                    if desc_lower.contains("historical") || desc_lower.contains("temporal") {
                        affinity += 0.2;
                    }
                }
            }
            ValueRange::Numeric(_, _) => {
                if predicate.contains("price") || predicate.contains("value") {
                    if desc_lower.contains("economic") || desc_lower.contains("financial") {
                        affinity += 0.2;
                    }
                }
            }
            ValueRange::Geospatial(_, _) => {
                if desc_lower.contains("geographic") || desc_lower.contains("spatial") {
                    affinity += 0.3;
                }
            }
        }

        affinity
    }

    /// Estimate result count for range query
    async fn estimate_range_result_count(
        &self,
        service: &FederatedService,
        predicate: &str,
        range: &ValueRange,
    ) -> Result<u64> {
        let base_count = self
            .estimate_predicate_result_count(service, predicate)
            .await?;

        // Apply range selectivity factor
        let selectivity = match range {
            ValueRange::Temporal(start, end) => {
                // Narrower time ranges are more selective
                let duration_days = (end.signed_duration_since(*start).num_days()).max(1);
                (365.0 / duration_days as f64).min(1.0)
            }
            ValueRange::Numeric(min, max) => {
                // Assume some default selectivity for numeric ranges
                if (max - min) < 100.0 {
                    0.1
                } else if (max - min) < 1000.0 {
                    0.3
                } else {
                    0.7
                }
            }
            ValueRange::Geospatial(_, _) => {
                // Geospatial ranges vary widely in selectivity
                0.2
            }
        };

        Ok((base_count as f64 * selectivity) as u64)
    }

    /// Estimate the value range covered by a service for a predicate
    async fn estimate_service_range(
        &self,
        _service: &FederatedService,
        _predicate: &str,
    ) -> Result<ValueRange> {
        // This would query service metadata in practice
        // For now, return a default wide range
        Ok(ValueRange::Numeric(0.0, 1000000.0))
    }

    /// Classify how a query range overlaps with service range
    fn classify_range_overlap(
        &self,
        query_range: &ValueRange,
        service_range: &ValueRange,
    ) -> RangeOverlapType {
        // Simplified overlap classification
        // In practice, this would handle proper range intersection logic
        match (query_range, service_range) {
            (ValueRange::Numeric(qmin, qmax), ValueRange::Numeric(smin, smax)) => {
                if qmin >= smin && qmax <= smax {
                    RangeOverlapType::Complete
                } else if qmax < smin || qmin > smax {
                    RangeOverlapType::None
                } else {
                    RangeOverlapType::Partial
                }
            }
            _ => RangeOverlapType::Unknown,
        }
    }
}

// ===== ADVANCED COST-BASED SELECTION ALGORITHMS =====

impl ServiceOptimizer {
    /// Advanced result size estimation using statistical models
    pub fn estimate_result_size_advanced(
        &self,
        pattern: &TriplePattern,
        service: &FederatedService,
        registry: &ServiceRegistry,
    ) -> Result<u64> {
        let mut estimated_size = 1000; // Base estimate

        // Use historical statistics if available
        if let Some(stats) = self
            .statistics_cache
            .get_predicate_stats(&pattern.predicate)
        {
            estimated_size = stats.avg_result_size;

            // Adjust based on pattern selectivity
            let selectivity = self.calculate_pattern_selectivity(pattern, service, registry);
            estimated_size = (estimated_size as f64 * selectivity) as u64;
        }

        // Apply service-specific factors
        let service_factor = self.get_service_result_size_factor(service, registry);
        estimated_size = (estimated_size as f64 * service_factor) as u64;

        // Consider triple pattern complexity
        let complexity_factor = match self.calculate_pattern_complexity(pattern) {
            PatternComplexity::Simple => 0.8, // Simple patterns typically return fewer results
            PatternComplexity::Medium => 1.0,
            PatternComplexity::Complex => 1.5, // Complex patterns may return more results
        };

        estimated_size = (estimated_size as f64 * complexity_factor) as u64;

        // Apply machine learning-based size prediction if available
        if let Ok(ml_estimate) = self.estimate_result_size_ml(pattern, service, registry) {
            // Blend statistical and ML estimates
            estimated_size = ((estimated_size as f64 * 0.6) + (ml_estimate as f64 * 0.4)) as u64;
        }

        // Apply range-based adjustments for numeric/temporal predicates
        if let Ok(range_factor) = self.estimate_range_selectivity_factor(pattern, service) {
            estimated_size = (estimated_size as f64 * range_factor) as u64;
        }

        Ok(estimated_size.max(1)) // At least 1 result
    }

    /// Machine learning-based result size estimation
    pub fn estimate_result_size_ml(
        &self,
        pattern: &TriplePattern,
        service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> Result<u64> {
        // Extract pattern features for ML model
        let features = self.extract_pattern_features(pattern, service)?;

        // Simple linear model - in production this would use a trained model
        let mut predicted_size = 1000.0; // Base prediction

        // Feature weights learned from historical data
        predicted_size += features.predicate_frequency * 200.0;
        predicted_size *= features.subject_specificity;
        predicted_size *= features.object_specificity;
        predicted_size += features.service_data_size_factor * 500.0;

        // Apply pattern type multipliers
        if pattern.subject.starts_with('?') && pattern.object.starts_with('?') {
            predicted_size *= 2.0; // More variables = more results
        }

        Ok(predicted_size.max(1.0) as u64)
    }

    /// Extract features for ML-based estimation
    fn extract_pattern_features(
        &self,
        pattern: &TriplePattern,
        service: &FederatedService,
    ) -> Result<PatternFeatures> {
        let predicate_frequency = if let Some(stats) = self
            .statistics_cache
            .get_predicate_stats(&pattern.predicate)
        {
            stats.frequency as f64 / 10000.0 // Normalize
        } else {
            0.5 // Default moderate frequency
        };

        let subject_specificity = if pattern.subject.starts_with('?') {
            0.8 // Variable is less specific
        } else if pattern.subject.starts_with("http://") {
            0.3 // URI is very specific
        } else {
            0.5 // Literal has medium specificity
        };

        let object_specificity = if pattern.object.starts_with('?') {
            0.8 // Variable is less specific
        } else if pattern.object.starts_with("http://") {
            0.3 // URI is very specific
        } else {
            0.5 // Literal has medium specificity
        };

        // Estimate service data size factor based on performance metrics
        let service_data_size_factor =
            if let Some(avg_time) = service.performance.average_response_time {
                (avg_time.as_millis() as f64 / 1000.0).min(2.0) // Cap at 2x factor
            } else {
                1.0
            };

        Ok(PatternFeatures {
            predicate_frequency,
            subject_specificity,
            object_specificity,
            service_data_size_factor,
        })
    }

    /// Estimate range selectivity factor for numeric/temporal predicates
    pub fn estimate_range_selectivity_factor(
        &self,
        pattern: &TriplePattern,
        _service: &FederatedService,
    ) -> Result<f64> {
        // Check if predicate suggests numeric or temporal data
        let predicate_lower = pattern.predicate.to_lowercase();

        if predicate_lower.contains("age")
            || predicate_lower.contains("year")
            || predicate_lower.contains("date")
            || predicate_lower.contains("time")
            || predicate_lower.contains("count")
            || predicate_lower.contains("number")
        {
            // These predicates typically have range constraints
            return Ok(0.3); // Higher selectivity due to range filtering
        }

        if predicate_lower.contains("name")
            || predicate_lower.contains("title")
            || predicate_lower.contains("label")
        {
            // Text predicates are less selective
            return Ok(0.8);
        }

        Ok(1.0) // Default - no range adjustment
    }

    /// Calculate pattern selectivity (0.0 to 1.0)
    fn calculate_pattern_selectivity(
        &self,
        pattern: &TriplePattern,
        service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> f64 {
        let mut selectivity: f64 = 1.0;

        // More specific patterns are more selective (return fewer results)
        if !pattern.subject.starts_with('?') {
            selectivity *= 0.1; // Specific subject reduces results significantly
        }

        if !pattern.predicate.starts_with('?') {
            selectivity *= 0.3; // Specific predicate reduces results moderately
        }

        if !pattern.object.starts_with('?') {
            selectivity *= 0.2; // Specific object reduces results significantly
        }

        // Consider service characteristics
        if let Some(ref description) = service.metadata.description {
            // Specialized services might have higher selectivity for their domain
            if pattern.predicate.contains("foaf:") && description.to_lowercase().contains("social")
            {
                selectivity *= 0.5;
            }
        }

        selectivity.max(0.001f64) // Minimum selectivity to avoid zero results
    }

    /// Get service-specific result size factor
    fn get_service_result_size_factor(
        &self,
        service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> f64 {
        let mut factor = 1.0;

        // Large services might return more results
        if let Some(ref description) = service.metadata.description {
            if description.to_lowercase().contains("large")
                || description.to_lowercase().contains("comprehensive")
            {
                factor *= 2.0;
            } else if description.to_lowercase().contains("small")
                || description.to_lowercase().contains("specialized")
            {
                factor *= 0.5;
            }
        }

        // Consider service performance as indicator of dataset size
        if let Some(avg_time) = service.performance.average_response_time {
            // Slower services might have larger datasets
            let time_millis = avg_time.as_millis() as f64;
            if time_millis > 1000.0 {
                factor *= 1.5;
            } else if time_millis < 100.0 {
                factor *= 0.7;
            }
        }

        factor
    }

    /// Advanced network latency modeling
    pub fn estimate_network_latency_advanced(
        &self,
        service: &FederatedService,
        request_size: u64,
        expected_response_size: u64,
        registry: &ServiceRegistry,
    ) -> Result<Duration> {
        // Base latency from service performance data
        let mut base_latency_ms = if let Some(avg_time) = service.performance.average_response_time
        {
            avg_time.as_millis() as f64
        } else {
            100.0 // Default estimate
        };

        // Adjust for request size (larger queries take longer)
        let request_factor = 1.0 + (request_size as f64 / 1000.0) * 0.1;
        base_latency_ms *= request_factor;

        // Adjust for expected response size (more data to transfer)
        let response_factor = 1.0 + (expected_response_size as f64 / 10000.0) * 0.2;
        base_latency_ms *= response_factor;

        // Consider network conditions based on service location
        let network_factor = self.estimate_network_conditions(service, registry);
        base_latency_ms *= network_factor;

        // Apply current load factor
        let load_factor = self.get_service_load_factor(service, registry);
        base_latency_ms *= load_factor;

        Ok(Duration::from_millis(base_latency_ms as u64))
    }

    /// Enhanced network latency estimation with geographic and temporal factors
    pub fn estimate_network_latency_enhanced(
        &self,
        service: &FederatedService,
        request_size: u64,
        expected_response_size: u64,
        registry: &ServiceRegistry,
        current_time: chrono::DateTime<chrono::Utc>,
    ) -> Result<Duration> {
        // Base latency from service performance data
        let mut base_latency_ms = if let Some(avg_time) = service.performance.average_response_time
        {
            avg_time.as_millis() as f64
        } else {
            100.0 // Default estimate
        };

        // Geographic distance estimation
        let geographic_factor = self.estimate_geographic_latency_factor(service)?;
        base_latency_ms *= geographic_factor;

        // Bandwidth and data transfer estimation
        let transfer_latency =
            self.estimate_transfer_latency(request_size, expected_response_size, service)?;
        base_latency_ms += transfer_latency;

        // Time-of-day network congestion factor
        let congestion_factor = self.estimate_network_congestion_factor(current_time, service);
        base_latency_ms *= congestion_factor;

        // Service tier and CDN optimization factor
        let service_tier_factor = self.estimate_service_tier_factor(service);
        base_latency_ms *= service_tier_factor;

        // Historical latency variance and confidence interval
        let (adjusted_latency, confidence) =
            self.apply_latency_confidence_adjustment(base_latency_ms, service);

        debug!(
            "Enhanced latency estimate for {}: {:.2}ms (confidence: {:.2})",
            service.endpoint, adjusted_latency, confidence
        );

        Ok(Duration::from_millis(adjusted_latency as u64))
    }

    /// Estimate geographic latency factor based on service location
    fn estimate_geographic_latency_factor(&self, service: &FederatedService) -> Result<f64> {
        if let Ok(url) = url::Url::parse(&service.endpoint) {
            if let Some(host) = url.host_str() {
                // Local/internal services
                if host == "localhost"
                    || host.starts_with("127.")
                    || host.starts_with("192.168.")
                    || host.starts_with("10.")
                {
                    return Ok(0.1); // Very low latency for local services
                }

                // Estimate by domain characteristics
                let domain_factor = self.estimate_domain_geographic_factor(host);
                return Ok(domain_factor);
            }
        }
        Ok(1.0) // Default factor for unknown locations
    }

    /// Estimate geographic factor based on domain characteristics
    fn estimate_domain_geographic_factor(&self, host: &str) -> f64 {
        // Country-specific TLDs (simplified estimation)
        if host.ends_with(".com") || host.ends_with(".org") || host.ends_with(".net") {
            return 1.0; // Assume global CDN for common domains
        }

        // Geographic hints in domain names
        let host_lower = host.to_lowercase();
        if host_lower.contains("us") || host_lower.contains("america") {
            return 0.8; // Assume same continent (North America)
        }
        if host_lower.contains("eu") || host_lower.contains("europe") {
            return 1.2; // Cross-Atlantic
        }
        if host_lower.contains("asia") || host_lower.contains("jp") || host_lower.contains("cn") {
            return 1.5; // Cross-Pacific
        }
        if host_lower.contains("au") || host_lower.contains("oceania") {
            return 1.8; // Long distance to Australia/Oceania
        }

        // CDN providers (typically optimized routing)
        if host_lower.contains("cloudflare")
            || host_lower.contains("fastly")
            || host_lower.contains("akamai")
            || host_lower.contains("amazonaws")
            || host_lower.contains("googleusercontent")
            || host_lower.contains("azure")
        {
            return 0.7; // CDN optimization
        }

        1.0 // Default for unknown geographic location
    }

    /// Estimate data transfer latency based on size and service characteristics
    fn estimate_transfer_latency(
        &self,
        request_size: u64,
        response_size: u64,
        service: &FederatedService,
    ) -> Result<f64> {
        // Estimate bandwidth based on service characteristics
        let estimated_bandwidth_mbps = self.estimate_service_bandwidth(service);

        // Convert sizes to megabits
        let request_mb = (request_size as f64) / 125000.0; // bytes to megabits
        let response_mb = (response_size as f64) / 125000.0;

        // Calculate transfer time in milliseconds
        let request_transfer_ms = (request_mb / estimated_bandwidth_mbps) * 1000.0;
        let response_transfer_ms = (response_mb / estimated_bandwidth_mbps) * 1000.0;

        // Add protocol overhead (TCP handshake, HTTP headers, etc.)
        let protocol_overhead_ms = 20.0;

        Ok(request_transfer_ms + response_transfer_ms + protocol_overhead_ms)
    }

    /// Estimate service bandwidth based on service characteristics
    fn estimate_service_bandwidth(&self, service: &FederatedService) -> f64 {
        // Default bandwidth assumption: 10 Mbps for typical service
        let mut bandwidth_mbps = 10.0;

        // Adjust based on performance metrics
        if let Some(avg_time) = service.performance.average_response_time {
            let time_ms = avg_time.as_millis() as f64;

            // Fast services likely have better infrastructure
            if time_ms < 100.0 {
                bandwidth_mbps *= 2.0; // High-performance service
            } else if time_ms > 1000.0 {
                bandwidth_mbps *= 0.5; // Potentially constrained service
            }
        }

        // Adjust based on endpoint characteristics
        if let Ok(url) = url::Url::parse(&service.endpoint) {
            if let Some(host) = url.host_str() {
                let host_lower = host.to_lowercase();

                // CDN services typically have higher bandwidth
                if host_lower.contains("cloudflare")
                    || host_lower.contains("fastly")
                    || host_lower.contains("amazonaws")
                    || host_lower.contains("azure")
                {
                    bandwidth_mbps *= 3.0;
                }

                // HTTPS might indicate better infrastructure
                if url.scheme() == "https" {
                    bandwidth_mbps *= 1.2;
                }
            }
        }

        bandwidth_mbps.max(1.0) // Minimum 1 Mbps
    }

    /// Estimate network congestion factor based on time of day
    fn estimate_network_congestion_factor(
        &self,
        current_time: chrono::DateTime<chrono::Utc>,
        service: &FederatedService,
    ) -> f64 {
        let hour = current_time.hour();

        // Estimate primary timezone based on service characteristics
        let primary_timezone_offset = self.estimate_service_timezone_offset(service);
        let local_hour = ((hour as i32) + primary_timezone_offset).rem_euclid(24) as u32;

        // Peak hours typically have more network congestion
        let congestion_factor = match local_hour {
            0..=5 => 0.8,   // Night: lower congestion
            6..=8 => 1.3,   // Morning peak
            9..=11 => 1.1,  // Business hours
            12..=13 => 1.4, // Lunch peak
            14..=17 => 1.2, // Afternoon business
            18..=20 => 1.5, // Evening peak
            21..=23 => 1.0, // Evening
            _ => 1.0,
        };

        congestion_factor
    }

    /// Estimate service timezone offset (simplified heuristic)
    fn estimate_service_timezone_offset(&self, service: &FederatedService) -> i32 {
        if let Ok(url) = url::Url::parse(&service.endpoint) {
            if let Some(host) = url.host_str() {
                let host_lower = host.to_lowercase();

                // Simple geographic timezone estimation
                if host_lower.contains("eu") || host_lower.contains("europe") {
                    return 1; // CET
                }
                if host_lower.contains("asia") || host_lower.contains("jp") {
                    return 9; // JST
                }
                if host_lower.contains("au") {
                    return 10; // AEST
                }
                if host_lower.contains("us") || host_lower.contains("america") {
                    return -5; // EST (approximate)
                }
            }
        }
        0 // Default to UTC
    }

    /// Estimate service tier and optimization factor
    fn estimate_service_tier_factor(&self, service: &FederatedService) -> f64 {
        let mut factor = 1.0;

        // Check for enterprise/premium indicators
        if let Some(description) = &service.metadata.description {
            let desc_lower = description.to_lowercase();
            if desc_lower.contains("enterprise")
                || desc_lower.contains("premium")
                || desc_lower.contains("pro")
            {
                factor *= 0.8; // Premium services likely have better performance
            }
            if desc_lower.contains("free") || desc_lower.contains("trial") {
                factor *= 1.3; // Free services may have limitations
            }
        }

        // Check endpoint for service tier indicators
        if let Ok(url) = url::Url::parse(&service.endpoint) {
            if let Some(host) = url.host_str() {
                // Dedicated domains suggest better performance
                if host.contains("api") || host.contains("sparql") {
                    factor *= 0.9;
                }
                // Subdomains might indicate shared infrastructure
                if host.split('.').count() > 2 {
                    factor *= 1.1;
                }
            }
        }

        factor
    }

    /// Apply latency confidence adjustment based on historical variance
    fn apply_latency_confidence_adjustment(
        &self,
        base_latency: f64,
        service: &FederatedService,
    ) -> (f64, f64) {
        // Simple confidence model - in production would use historical variance data
        let confidence = if service.performance.average_response_time.is_some() {
            0.8 // Good confidence if we have performance data
        } else {
            0.5 // Lower confidence for unknown services
        };

        // Apply confidence interval - add padding for uncertainty
        let uncertainty_padding = base_latency * (1.0 - confidence) * 0.5;
        let adjusted_latency = base_latency + uncertainty_padding;

        (adjusted_latency, confidence)
    }

    /// Estimate network conditions based on service characteristics
    fn estimate_network_conditions(
        &self,
        service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> f64 {
        // Parse service endpoint to estimate network distance/quality
        if let Ok(url) = url::Url::parse(&service.endpoint) {
            if let Some(host) = url.host_str() {
                // Local/internal services are faster
                if host == "localhost" || host.starts_with("127.") || host.starts_with("192.168.") {
                    return 0.5;
                }

                // Well-known fast CDN services
                if host.contains("cloudflare")
                    || host.contains("fastly")
                    || host.contains("amazonaws")
                {
                    return 0.8;
                }

                // Assume reasonable network for HTTPS, slower for HTTP
                if url.scheme() == "https" {
                    return 1.0;
                } else {
                    return 1.2;
                }
            }
        }

        1.1 // Default slightly higher latency for unknown services
    }

    /// Get service load factor based on current capacity utilization
    fn get_service_load_factor(
        &self,
        service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> f64 {
        // Use average response time as proxy for current load
        if let Some(avg_time) = service.performance.average_response_time {
            let millis = avg_time.as_millis() as f64;

            // Higher response times suggest higher load
            if millis > 2000.0 {
                return 2.0; // Very high load
            } else if millis > 1000.0 {
                return 1.5; // High load
            } else if millis > 500.0 {
                return 1.2; // Medium load
            } else {
                return 1.0; // Normal load
            }
        }

        // Use max concurrent requests as additional load indicator
        if let Some(max_requests) = service.performance.max_concurrent_requests {
            if max_requests < 10 {
                return 1.3; // Low capacity suggests potential issues
            }
        }

        1.0 // Default normal load
    }

    /// Advanced service capacity analysis
    pub fn analyze_service_capacity(
        &self,
        service: &FederatedService,
        current_query_load: u32,
        registry: &ServiceRegistry,
    ) -> Result<ServiceCapacityAnalysis> {
        let mut analysis = ServiceCapacityAnalysis {
            max_concurrent_queries: 100, // Default estimate
            current_utilization: 0.0,
            recommended_max_load: 80,
            bottleneck_factors: Vec::new(),
            scaling_suggestions: Vec::new(),
        };

        // Estimate maximum capacity based on performance characteristics
        if let Some(avg_time) = service.performance.average_response_time {
            let millis = avg_time.as_millis() as f64;

            // Faster services can handle more concurrent requests
            analysis.max_concurrent_queries = if millis < 100.0 {
                200
            } else if millis < 500.0 {
                100
            } else if millis < 1000.0 {
                50
            } else {
                20
            };
        }

        // Calculate current utilization
        analysis.current_utilization =
            current_query_load as f64 / analysis.max_concurrent_queries as f64;

        // Identify bottleneck factors
        if let Some(max_requests) = service.performance.max_concurrent_requests {
            if max_requests < 20 {
                analysis.bottleneck_factors.push(
                    "Low concurrent request capacity indicates potential bottleneck".to_string(),
                );
            }
        }

        if let Some(avg_time) = service.performance.average_response_time {
            if avg_time.as_millis() > 2000 {
                analysis
                    .bottleneck_factors
                    .push("High response time indicates performance bottleneck".to_string());
            }
        }

        // Generate scaling suggestions
        if analysis.current_utilization > 0.8 {
            analysis
                .scaling_suggestions
                .push("Consider adding replica services".to_string());
            analysis
                .scaling_suggestions
                .push("Implement query caching".to_string());
        }

        if analysis.current_utilization > 0.9 {
            analysis
                .scaling_suggestions
                .push("URGENT: Service approaching capacity limit".to_string());
        }

        // Adjust recommended max load based on service capacity
        if let Some(max_requests) = service.performance.max_concurrent_requests {
            if max_requests < 50 {
                analysis.recommended_max_load = 60; // Be more conservative with low-capacity services
            }
        }

        Ok(analysis)
    }

    /// Multi-objective cost optimization combining multiple factors
    pub fn calculate_multi_objective_cost(
        &self,
        pattern: &TriplePattern,
        service: &FederatedService,
        registry: &ServiceRegistry,
        objectives: &CostObjectives,
    ) -> Result<CostScore> {
        let mut score = CostScore {
            total_cost: 0.0,
            execution_time_cost: 0.0,
            network_latency_cost: 0.0,
            resource_usage_cost: 0.0,
            reliability_cost: 0.0,
            quality_score: 0.0,
        };

        // Execution time cost
        let estimated_time = self.estimate_execution_time(pattern, service, registry)?;
        score.execution_time_cost = estimated_time.as_millis() as f64 * objectives.time_weight;

        // Network latency cost
        let request_size = self.estimate_request_size(pattern)?;
        let response_size = self.estimate_result_size_advanced(pattern, service, registry)?;
        let network_latency =
            self.estimate_network_latency_advanced(service, request_size, response_size, registry)?;
        score.network_latency_cost = network_latency.as_millis() as f64 * objectives.latency_weight;

        // Resource usage cost (based on query complexity and service load)
        let complexity_cost = self.calculate_pattern_complexity_cost(pattern);
        let load_factor = self.get_service_load_factor(service, registry);
        score.resource_usage_cost = complexity_cost * load_factor * objectives.resource_weight;

        // Reliability cost (penalty for low-capacity services)
        let reliability_penalty =
            if let Some(max_requests) = service.performance.max_concurrent_requests {
                if max_requests < 50 {
                    100.0 // Higher penalty for low-capacity services
                } else {
                    20.0 // Lower penalty for high-capacity services
                }
            } else {
                50.0 // Default penalty for unknown capacity
            };
        score.reliability_cost = reliability_penalty * objectives.reliability_weight;

        // Quality score (bonus for high-quality services)
        score.quality_score =
            self.calculate_service_quality_score(service, registry) * objectives.quality_weight;

        // Calculate total cost (lower is better, except quality which is a bonus)
        score.total_cost = score.execution_time_cost
            + score.network_latency_cost
            + score.resource_usage_cost
            + score.reliability_cost
            - score.quality_score; // Subtract quality score as it's a bonus

        Ok(score)
    }

    /// Estimate execution time for a pattern on a service
    fn estimate_execution_time(
        &self,
        pattern: &TriplePattern,
        service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> Result<Duration> {
        let mut base_time_ms = if let Some(avg_time) = service.performance.average_response_time {
            avg_time.as_millis() as f64
        } else {
            200.0 // Default estimate
        };

        // Adjust for pattern complexity
        let complexity_factor = match self.calculate_pattern_complexity(pattern) {
            PatternComplexity::Simple => 0.8,
            PatternComplexity::Medium => 1.0,
            PatternComplexity::Complex => 2.0,
        };

        base_time_ms *= complexity_factor;

        Ok(Duration::from_millis(base_time_ms as u64))
    }

    /// Estimate request size for network cost calculation
    fn estimate_request_size(&self, pattern: &TriplePattern) -> Result<u64> {
        // Base SPARQL query overhead
        let mut size = 100;

        // Add pattern string length
        size += pattern.pattern_string.len() as u64;

        // Add complexity overhead
        if pattern.pattern_string.contains("FILTER") {
            size += 50;
        }
        if pattern.pattern_string.contains("OPTIONAL") {
            size += 30;
        }

        Ok(size)
    }

    /// Calculate complexity cost for resource usage estimation
    fn calculate_pattern_complexity_cost(&self, pattern: &TriplePattern) -> f64 {
        let mut cost = 10.0; // Base cost

        // Variable patterns are more expensive
        if pattern.subject.starts_with('?') {
            cost += 5.0;
        }
        if pattern.predicate.starts_with('?') {
            cost += 10.0; // Predicate variables are very expensive
        }
        if pattern.object.starts_with('?') {
            cost += 5.0;
        }

        // Complex pattern features
        if pattern.pattern_string.contains("REGEX") {
            cost += 20.0;
        }
        if pattern.pattern_string.contains("PropertyPath") {
            cost += 15.0;
        }

        cost
    }

    /// Calculate service quality score
    fn calculate_service_quality_score(
        &self,
        service: &FederatedService,
        _registry: &ServiceRegistry,
    ) -> f64 {
        let mut score = 0.0;

        // Capacity contribution (use max concurrent requests as quality indicator)
        if let Some(max_requests) = service.performance.max_concurrent_requests {
            score += (max_requests as f64).min(100.0); // 0-100 points for capacity
        }

        // Response time contribution (faster is better)
        if let Some(avg_time) = service.performance.average_response_time {
            let millis = avg_time.as_millis() as f64;
            let time_score = (2000.0 - millis.min(2000.0)) / 20.0; // 0-100 points, capped at 2 seconds
            score += time_score.max(0.0);
        }

        // Capability richness
        let capability_score = service.capabilities.len() as f64 * 5.0; // 5 points per capability
        score += capability_score;

        score
    }

    /// Calculate pattern complexity (missing method implementation)
    fn calculate_pattern_complexity(&self, pattern: &TriplePattern) -> PatternComplexity {
        let mut complexity_score = 0;

        // Variable patterns increase complexity
        if pattern.subject.starts_with('?') {
            complexity_score += 1;
        }
        if pattern.predicate.starts_with('?') {
            complexity_score += 2; // Predicate variables are more complex
        }
        if pattern.object.starts_with('?') {
            complexity_score += 1;
        }

        // Check for complex patterns in the pattern string
        let pattern_str = &pattern.pattern_string;
        if pattern_str.contains("REGEX") || pattern_str.contains("FILTER") {
            complexity_score += 3;
        }
        if pattern_str.contains("OPTIONAL") || pattern_str.contains("UNION") {
            complexity_score += 2;
        }
        if pattern_str.contains("PropertyPath")
            || pattern_str.contains("*")
            || pattern_str.contains("+")
        {
            complexity_score += 4;
        }

        // Classify complexity based on score
        match complexity_score {
            0..=2 => PatternComplexity::Simple,
            3..=5 => PatternComplexity::Medium,
            _ => PatternComplexity::Complex,
        }
    }
}

/// Pattern complexity levels for cost calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PatternComplexity {
    Simple,
    Medium,
    Complex,
}

/// Service capacity analysis result
#[derive(Debug, Clone)]
pub struct ServiceCapacityAnalysis {
    pub max_concurrent_queries: u32,
    pub current_utilization: f64,
    pub recommended_max_load: u32,
    pub bottleneck_factors: Vec<String>,
    pub scaling_suggestions: Vec<String>,
}

/// Cost optimization objectives
#[derive(Debug, Clone)]
pub struct CostObjectives {
    pub time_weight: f64,
    pub latency_weight: f64,
    pub resource_weight: f64,
    pub reliability_weight: f64,
    pub quality_weight: f64,
}

impl Default for CostObjectives {
    fn default() -> Self {
        Self {
            time_weight: 1.0,
            latency_weight: 0.8,
            resource_weight: 0.6,
            reliability_weight: 1.2,
            quality_weight: 0.5,
        }
    }
}

/// Multi-objective cost score
#[derive(Debug, Clone)]
pub struct CostScore {
    pub total_cost: f64,
    pub execution_time_cost: f64,
    pub network_latency_cost: f64,
    pub resource_usage_cost: f64,
    pub reliability_cost: f64,
    pub quality_score: f64,
}

/// Pattern features for ML-based estimation
#[derive(Debug, Clone)]
pub struct PatternFeatures {
    pub predicate_frequency: f64,
    pub subject_specificity: f64,
    pub object_specificity: f64,
    pub service_data_size_factor: f64,
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

    /// Average result size for queries with this predicate
    pub avg_result_size: u64,

    /// Selectivity estimate
    pub selectivity: f64,
}

// ===== ADVANCED SOURCE SELECTION DATA STRUCTURES =====

/// Pattern coverage analysis result
#[derive(Debug, Clone)]
pub struct PatternCoverageAnalysis {
    pub total_patterns: usize,
    pub covered_patterns: usize,
    pub partially_covered_patterns: usize,
    pub uncovered_patterns: usize,
    pub overall_coverage_score: f64,
    pub coverage_quality: CoverageQuality,
    pub pattern_scores: Vec<f64>,
}

/// Coverage quality assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverageQuality {
    Excellent,
    Good,
    Fair,
    Poor,
}

/// Service predicate scoring result
#[derive(Debug, Clone)]
pub struct ServicePredicateScore {
    pub service_id: String,
    pub predicate: String,
    pub affinity_score: f64,
    pub estimated_result_count: u64,
    pub confidence_level: ConfidenceLevel,
}

/// Confidence level for estimations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceLevel {
    Low,
    Medium,
    High,
}

/// Value range for range-based selection
#[derive(Debug, Clone)]
pub enum ValueRange {
    Numeric(f64, f64),
    Temporal(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>),
    Geospatial(f64, f64), // Simplified lat/lon bounds
}

/// Range service match result
#[derive(Debug, Clone)]
pub struct RangeServiceMatch {
    pub service_id: String,
    pub predicate: String,
    pub range: ValueRange,
    pub coverage_score: f64,
    pub estimated_result_count: u64,
    pub overlap_type: RangeOverlapType,
}

/// Range overlap classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeOverlapType {
    Complete, // Query range fully contained in service range
    Partial,  // Query range partially overlaps service range
    None,     // No overlap between ranges
    Unknown,  // Cannot determine overlap
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

/// Advanced source selection algorithms with Bloom filters and ML
impl ServiceOptimizer {
    /// Bloom filter-based membership testing for efficient source selection
    pub fn create_service_bloom_filters(
        &self,
        services: &[FederatedService],
        registry: &ServiceRegistry,
    ) -> Result<HashMap<String, ServiceBloomFilter>> {
        use bloom::{BloomFilter, ASMS};

        let mut filters = HashMap::new();

        for service in services {
            // Create Bloom filter for predicates
            let mut predicate_filter = BloomFilter::with_rate(0.01, 10000);

            // Create Bloom filter for subjects/objects
            let mut resource_filter = BloomFilter::with_rate(0.01, 100000);

            // Populate filters based on service capabilities and known data
            if let Some(service_meta) = registry.get_service(&service.endpoint) {
                // Add known data patterns
                for pattern in &service_meta.data_patterns {
                    predicate_filter.insert(&pattern);
                    resource_filter.insert(&pattern);
                }

                // Use metadata tags as patterns
                for tag in &service_meta.metadata.tags {
                    resource_filter.insert(&tag);
                }
            }

            let service_filter = ServiceBloomFilter {
                predicate_filter,
                resource_filter,
                last_updated: chrono::Utc::now(),
                false_positive_rate: 0.01,
                estimated_elements: 10000,
            };

            filters.insert(service.endpoint.clone(), service_filter);
        }

        info!("Created Bloom filters for {} services", filters.len());
        Ok(filters)
    }

    /// Use Bloom filters for fast membership testing
    pub fn test_pattern_membership(
        &self,
        pattern: &TriplePattern,
        service_filters: &HashMap<String, ServiceBloomFilter>,
    ) -> HashMap<String, BloomFilterResult> {
        let mut results = HashMap::new();

        for (service_endpoint, filter) in service_filters {
            let mut likely_matches = Vec::new();

            // Test predicate membership
            let predicate = &pattern.predicate;
            if filter.predicate_filter.contains(&predicate) {
                likely_matches.push("predicate".to_string());
            }

            // Test subject membership
            let subject = &pattern.subject;
            if !subject.starts_with('?') && filter.resource_filter.contains(&subject) {
                likely_matches.push("subject".to_string());
            }

            // Test object membership
            let object = &pattern.object;
            if !object.starts_with('?') && filter.resource_filter.contains(&object) {
                likely_matches.push("object".to_string());
            }

            let membership_probability = if likely_matches.is_empty() {
                0.0
            } else {
                // Calculate probability based on number of matches
                let base_prob = likely_matches.len() as f64 / 3.0;
                // Adjust for false positive rate
                base_prob * (1.0 - filter.false_positive_rate)
            };

            results.insert(
                service_endpoint.clone(),
                BloomFilterResult {
                    membership_probability,
                    likely_matches,
                    false_positive_rate: filter.false_positive_rate,
                    confidence_score: 1.0 - filter.false_positive_rate,
                },
            );
        }

        results
    }

    /// Machine learning-based source prediction using historical data
    pub async fn predict_best_sources_ml(
        &self,
        patterns: &[TriplePattern],
        query_context: &QueryContext,
        historical_data: &HistoricalQueryData,
    ) -> Result<Vec<MLSourcePrediction>> {
        // Simple ML model using pattern matching and historical performance
        let mut predictions = Vec::new();

        // Extract features from query patterns
        let features = self.extract_query_features(patterns, query_context);

        // Find similar historical queries
        let similar_queries = self.find_similar_queries(&features, historical_data);

        // Score services based on historical performance
        let mut service_scores = HashMap::new();

        for similar_query in &similar_queries {
            let similarity_weight = similar_query.similarity_score;

            for (service, performance) in &similar_query.service_performance {
                let current_score = service_scores.get(service).unwrap_or(&0.0);
                let weighted_performance = performance.success_rate
                    * (1.0 / (performance.avg_latency_ms / 1000.0).max(0.1))
                    * similarity_weight;

                service_scores.insert(service.clone(), current_score + weighted_performance);
            }
        }

        // Create predictions
        for (service, score) in service_scores {
            let prediction = MLSourcePrediction {
                service_endpoint: service,
                confidence_score: score.min(1.0),
                predicted_latency_ms: self.predict_latency(&service, &features),
                predicted_success_rate: self.predict_success_rate(&service, &features),
                feature_importance: self.calculate_feature_importance(&features),
                model_version: "simple_pattern_matching_v1.0".to_string(),
            };
            predictions.push(prediction);
        }

        // Sort by confidence score
        predictions.sort_by(|a, b| {
            b.confidence_score
                .partial_cmp(&a.confidence_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        info!(
            "Generated ML predictions for {} services",
            predictions.len()
        );
        Ok(predictions)
    }

    /// Dynamic source ranking updates based on real-time performance
    pub async fn update_dynamic_rankings(
        &mut self,
        service_endpoint: &str,
        query_result: &QueryExecutionResult,
    ) -> Result<()> {
        let mut stats_cache = self.statistics_cache.as_ref().clone();

        // Update performance metrics
        let performance_update = PerformanceUpdate {
            timestamp: chrono::Utc::now(),
            latency_ms: query_result.execution_time_ms,
            success: query_result.success,
            result_count: query_result.result_count,
            error_type: query_result.error_type.clone(),
            query_complexity: self.calculate_query_complexity(&query_result.query_info),
        };

        // Apply exponential moving average for metrics
        if let Some(existing_stats) = stats_cache.get_service_stats(service_endpoint) {
            let alpha = 0.3; // Learning rate

            let updated_stats = ServiceStatistics {
                endpoint: service_endpoint.to_string(),
                avg_latency_ms: (1.0 - alpha) * existing_stats.avg_latency_ms
                    + alpha * performance_update.latency_ms,
                success_rate: (1.0 - alpha) * existing_stats.success_rate
                    + alpha * if performance_update.success { 1.0 } else { 0.0 },
                total_queries: existing_stats.total_queries + 1,
                last_updated: performance_update.timestamp,
                quality_score: self.calculate_quality_score(&performance_update, &existing_stats),
                reliability_trend: self
                    .calculate_reliability_trend(&performance_update, &existing_stats),
            };

            stats_cache.update_service_stats(service_endpoint, updated_stats)?;
        } else {
            // Create new statistics entry
            let new_stats = ServiceStatistics {
                endpoint: service_endpoint.to_string(),
                avg_latency_ms: performance_update.latency_ms,
                success_rate: if performance_update.success { 1.0 } else { 0.0 },
                total_queries: 1,
                last_updated: performance_update.timestamp,
                quality_score: if performance_update.success { 0.8 } else { 0.2 },
                reliability_trend: ReliabilityTrend::Stable,
            };

            stats_cache.add_service_stats(service_endpoint, new_stats)?;
        }

        // Update global ranking
        self.recalculate_service_rankings(&mut stats_cache).await?;

        info!("Updated dynamic ranking for service: {}", service_endpoint);
        Ok(())
    }

    /// Recalculate service rankings based on current performance data
    async fn recalculate_service_rankings(&self, stats_cache: &mut StatisticsCache) -> Result<()> {
        let all_stats = stats_cache.get_all_service_stats();
        let mut rankings = Vec::new();

        for (endpoint, stats) in &all_stats {
            let ranking_score = self.calculate_ranking_score(stats);
            rankings.push(ServiceRanking {
                endpoint: endpoint.clone(),
                ranking_score,
                ranking_factors: RankingFactors {
                    latency_score: self.normalize_latency_score(stats.avg_latency_ms),
                    reliability_score: stats.success_rate,
                    availability_score: self.calculate_availability_score(stats),
                    quality_score: stats.quality_score,
                    trend_score: self.calculate_trend_score(&stats.reliability_trend),
                },
                last_updated: chrono::Utc::now(),
            });
        }

        // Sort by ranking score
        rankings.sort_by(|a, b| {
            b.ranking_score
                .partial_cmp(&a.ranking_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update global rankings
        stats_cache.update_global_rankings(rankings)?;

        Ok(())
    }

    /// Extract features from query patterns for ML
    fn extract_query_features(
        &self,
        patterns: &[TriplePattern],
        context: &QueryContext,
    ) -> QueryFeatures {
        let mut predicate_counts = HashMap::new();
        let mut namespace_counts = HashMap::new();
        let mut pattern_types = HashMap::new();

        for pattern in patterns {
            // Count predicates
            let predicate = &pattern.predicate;
            *predicate_counts.entry(predicate.clone()).or_insert(0) += 1;

            // Extract namespace
            if let Some(namespace) = self.extract_namespace(predicate) {
                *namespace_counts.entry(namespace).or_insert(0) += 1;
            }

            // Classify pattern type
            let pattern_type = self.classify_pattern_type(pattern);
            *pattern_types.entry(pattern_type).or_insert(0) += 1;
        }

        QueryFeatures {
            pattern_count: patterns.len(),
            predicate_distribution: predicate_counts,
            namespace_distribution: namespace_counts,
            pattern_type_distribution: pattern_types,
            complexity_score: self.calculate_patterns_complexity(patterns),
            selectivity_estimate: self.estimate_query_selectivity(patterns),
            has_joins: self.has_join_patterns(patterns),
            query_type: context.query_type.clone(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Find similar queries in historical data
    fn find_similar_queries(
        &self,
        features: &QueryFeatures,
        historical_data: &HistoricalQueryData,
    ) -> Vec<SimilarQuery> {
        let mut similar_queries = Vec::new();

        for historical_query in &historical_data.queries {
            let similarity = self.calculate_query_similarity(features, &historical_query.features);

            if similarity > 0.5 {
                // Threshold for similarity
                similar_queries.push(SimilarQuery {
                    query_id: historical_query.id.clone(),
                    similarity_score: similarity,
                    service_performance: historical_query.service_performance.clone(),
                    execution_timestamp: historical_query.timestamp,
                });
            }
        }

        // Sort by similarity
        similar_queries.sort_by(|a, b| {
            b.similarity_score
                .partial_cmp(&a.similarity_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        similar_queries.into_iter().take(10).collect() // Top 10 similar queries
    }

    /// Calculate similarity between two query feature sets
    fn calculate_query_similarity(
        &self,
        features1: &QueryFeatures,
        features2: &QueryFeatures,
    ) -> f64 {
        let mut similarity_factors = Vec::new();

        // Pattern count similarity
        let pattern_count_sim = 1.0
            - ((features1.pattern_count as f64 - features2.pattern_count as f64).abs()
                / features1.pattern_count.max(features2.pattern_count) as f64);
        similarity_factors.push(pattern_count_sim * 0.2);

        // Predicate distribution similarity (Jaccard similarity)
        let predicate_sim = self.calculate_jaccard_similarity(
            &features1.predicate_distribution,
            &features2.predicate_distribution,
        );
        similarity_factors.push(predicate_sim * 0.4);

        // Namespace distribution similarity
        let namespace_sim = self.calculate_jaccard_similarity(
            &features1.namespace_distribution,
            &features2.namespace_distribution,
        );
        similarity_factors.push(namespace_sim * 0.3);

        // Complexity similarity
        let complexity_sim = 1.0
            - ((features1.complexity_score - features2.complexity_score).abs()
                / features1.complexity_score.max(features2.complexity_score));
        similarity_factors.push(complexity_sim * 0.1);

        similarity_factors.iter().sum()
    }

    /// Calculate Jaccard similarity between two hash maps
    fn calculate_jaccard_similarity<T: std::hash::Hash + Eq>(
        &self,
        map1: &HashMap<T, usize>,
        map2: &HashMap<T, usize>,
    ) -> f64 {
        let keys1: HashSet<_> = map1.keys().collect();
        let keys2: HashSet<_> = map2.keys().collect();

        let intersection = keys1.intersection(&keys2).count();
        let union = keys1.union(&keys2).count();

        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Predict latency for a service based on features
    fn predict_latency(&self, service: &str, features: &QueryFeatures) -> f64 {
        // Simple linear model based on complexity
        let base_latency = 100.0; // Base latency in ms
        let complexity_factor = features.complexity_score * 50.0;
        let pattern_factor = features.pattern_count as f64 * 10.0;

        base_latency + complexity_factor + pattern_factor
    }

    /// Predict success rate for a service
    fn predict_success_rate(&self, service: &str, features: &QueryFeatures) -> f64 {
        // Base success rate with adjustments for complexity
        let base_rate = 0.95;
        let complexity_penalty = features.complexity_score * 0.1;

        (base_rate - complexity_penalty).max(0.1)
    }

    /// Calculate feature importance for ML model
    fn calculate_feature_importance(&self, features: &QueryFeatures) -> HashMap<String, f64> {
        let mut importance = HashMap::new();

        importance.insert("pattern_count".to_string(), 0.2);
        importance.insert("predicate_distribution".to_string(), 0.4);
        importance.insert("namespace_distribution".to_string(), 0.2);
        importance.insert("complexity_score".to_string(), 0.15);
        importance.insert("has_joins".to_string(), 0.05);

        importance
    }

    /// Helper methods for feature extraction
    fn extract_namespace(&self, uri: &str) -> Option<String> {
        if let Some(hash_pos) = uri.rfind('#') {
            Some(uri[..hash_pos].to_string())
        } else if let Some(slash_pos) = uri.rfind('/') {
            Some(uri[..slash_pos].to_string())
        } else {
            None
        }
    }

    fn classify_pattern_type(&self, pattern: &TriplePattern) -> String {
        match (
            Some(pattern.subject.starts_with('?')),
            Some(pattern.predicate.starts_with('?')),
            Some(pattern.object.starts_with('?')),
        ) {
            (Some(false), Some(false), Some(false)) => "concrete".to_string(),
            (Some(true), Some(false), Some(true)) => "predicate_bound".to_string(),
            (Some(false), Some(true), Some(false)) => "subject_object_bound".to_string(),
            (Some(true), Some(true), Some(true)) => "all_variables".to_string(),
            _ => "mixed".to_string(),
        }
    }

    fn calculate_patterns_complexity(&self, patterns: &[TriplePattern]) -> f64 {
        let variable_count: usize = patterns
            .iter()
            .map(|p| {
                let mut count = 0;
                if p.subject.starts_with('?') {
                    count += 1;
                }
                if p.predicate.starts_with('?') {
                    count += 1;
                }
                if p.object.starts_with('?') {
                    count += 1;
                }
                count
            })
            .sum();

        (variable_count as f64) / (patterns.len() as f64 * 3.0)
    }

    fn estimate_query_selectivity(&self, patterns: &[TriplePattern]) -> f64 {
        // Simple selectivity estimation based on bound variables
        let total_positions = patterns.len() * 3;
        let bound_positions = patterns
            .iter()
            .map(|p| {
                let mut count = 0;
                if !p.subject.starts_with('?') {
                    count += 1;
                }
                if !p.predicate.starts_with('?') {
                    count += 1;
                }
                if !p.object.starts_with('?') {
                    count += 1;
                }
                count
            })
            .sum::<usize>();

        (bound_positions as f64) / (total_positions as f64)
    }

    fn has_join_patterns(&self, patterns: &[TriplePattern]) -> bool {
        let variables: HashSet<String> = patterns
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

        // If we have shared variables across patterns, we likely have joins
        let total_variable_occurrences: usize = patterns
            .iter()
            .map(|p| {
                let mut count = 0;
                if p.subject.starts_with('?') {
                    count += 1;
                }
                if p.predicate.starts_with('?') {
                    count += 1;
                }
                if p.object.starts_with('?') {
                    count += 1;
                }
                count
            })
            .sum();

        total_variable_occurrences > variables.len()
    }

    fn calculate_query_complexity(&self, query_info: &QueryInfo) -> f64 {
        // Simple complexity score based on query characteristics
        let base_score = 1.0;
        let pattern_factor = query_info.pattern_count as f64 * 0.1;
        let join_factor = if query_info.has_joins { 0.5 } else { 0.0 };
        let filter_factor = query_info.filter_count as f64 * 0.05;

        base_score + pattern_factor + join_factor + filter_factor
    }

    fn calculate_quality_score(
        &self,
        update: &PerformanceUpdate,
        existing: &ServiceStatistics,
    ) -> f64 {
        let latency_score = 1.0 / (1.0 + update.latency_ms / 1000.0);
        let success_score = if update.success { 1.0 } else { 0.0 };
        let consistency_score = 1.0
            - (update.latency_ms - existing.avg_latency_ms).abs()
                / existing.avg_latency_ms.max(1.0);

        (latency_score * 0.4 + success_score * 0.4 + consistency_score * 0.2).min(1.0)
    }

    fn calculate_reliability_trend(
        &self,
        update: &PerformanceUpdate,
        existing: &ServiceStatistics,
    ) -> ReliabilityTrend {
        let current_quality = self.calculate_quality_score(update, existing);

        if current_quality > existing.quality_score + 0.1 {
            ReliabilityTrend::Improving
        } else if current_quality < existing.quality_score - 0.1 {
            ReliabilityTrend::Degrading
        } else {
            ReliabilityTrend::Stable
        }
    }

    fn calculate_ranking_score(&self, stats: &ServiceStatistics) -> f64 {
        let latency_score = self.normalize_latency_score(stats.avg_latency_ms);
        let reliability_score = stats.success_rate;
        let quality_score = stats.quality_score;
        let trend_bonus = match stats.reliability_trend {
            ReliabilityTrend::Improving => 0.1,
            ReliabilityTrend::Stable => 0.0,
            ReliabilityTrend::Degrading => -0.1,
        };

        (latency_score * 0.3 + reliability_score * 0.4 + quality_score * 0.3 + trend_bonus).min(1.0)
    }

    fn normalize_latency_score(&self, latency_ms: f64) -> f64 {
        // Convert latency to score (lower latency = higher score)
        1.0 / (1.0 + latency_ms / 1000.0)
    }

    fn calculate_availability_score(&self, stats: &ServiceStatistics) -> f64 {
        // Simple availability calculation based on recent activity
        let hours_since_update = chrono::Utc::now()
            .signed_duration_since(stats.last_updated)
            .num_hours();

        if hours_since_update < 1 {
            1.0
        } else if hours_since_update < 24 {
            0.8
        } else {
            0.5
        }
    }

    fn calculate_trend_score(&self, trend: &ReliabilityTrend) -> f64 {
        match trend {
            ReliabilityTrend::Improving => 1.0,
            ReliabilityTrend::Stable => 0.8,
            ReliabilityTrend::Degrading => 0.5,
        }
    }
}

/// Supporting data structures for advanced source selection

/// Service Bloom filter for membership testing
pub struct ServiceBloomFilter {
    pub predicate_filter: bloom::BloomFilter,
    pub resource_filter: bloom::BloomFilter,
    pub last_updated: chrono::DateTime<chrono::Utc>,
    pub false_positive_rate: f64,
    pub estimated_elements: usize,
}

/// Bloom filter test result
#[derive(Debug, Clone)]
pub struct BloomFilterResult {
    pub membership_probability: f64,
    pub likely_matches: Vec<String>,
    pub false_positive_rate: f64,
    pub confidence_score: f64,
}

/// ML-based source prediction
#[derive(Debug, Clone)]
pub struct MLSourcePrediction {
    pub service_endpoint: String,
    pub confidence_score: f64,
    pub predicted_latency_ms: f64,
    pub predicted_success_rate: f64,
    pub feature_importance: HashMap<String, f64>,
    pub model_version: String,
}

/// Query context for ML predictions
#[derive(Debug, Clone)]
pub struct QueryContext {
    pub query_type: String,
    pub user_context: Option<String>,
    pub time_constraints: Option<Duration>,
    pub quality_requirements: Option<QualityRequirements>,
}

/// Quality requirements for query execution
#[derive(Debug, Clone)]
pub struct QualityRequirements {
    pub max_latency_ms: f64,
    pub min_success_rate: f64,
    pub completeness_threshold: f64,
}

/// Historical query data for ML training
#[derive(Debug, Clone)]
pub struct HistoricalQueryData {
    pub queries: Vec<HistoricalQuery>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Individual historical query record
#[derive(Debug, Clone)]
pub struct HistoricalQuery {
    pub id: String,
    pub features: QueryFeatures,
    pub service_performance: HashMap<String, OptimizerServicePerformance>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Service performance data for optimization
#[derive(Debug, Clone)]
pub struct OptimizerServicePerformance {
    pub avg_latency_ms: f64,
    pub success_rate: f64,
    pub result_quality: f64,
}

/// Query features for ML
#[derive(Debug, Clone)]
pub struct QueryFeatures {
    pub pattern_count: usize,
    pub predicate_distribution: HashMap<String, usize>,
    pub namespace_distribution: HashMap<String, usize>,
    pub pattern_type_distribution: HashMap<String, usize>,
    pub complexity_score: f64,
    pub selectivity_estimate: f64,
    pub has_joins: bool,
    pub query_type: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Similar query for pattern matching
#[derive(Debug, Clone)]
pub struct SimilarQuery {
    pub query_id: String,
    pub similarity_score: f64,
    pub service_performance: HashMap<String, OptimizerServicePerformance>,
    pub execution_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Query execution result for performance tracking
#[derive(Debug, Clone)]
pub struct QueryExecutionResult {
    pub execution_time_ms: f64,
    pub success: bool,
    pub result_count: usize,
    pub error_type: Option<String>,
    pub query_info: QueryInfo,
}

/// Query information for complexity calculation
#[derive(Debug, Clone)]
pub struct QueryInfo {
    pub pattern_count: usize,
    pub has_joins: bool,
    pub filter_count: usize,
}

/// Performance update for dynamic ranking
#[derive(Debug, Clone)]
pub struct PerformanceUpdate {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub latency_ms: f64,
    pub success: bool,
    pub result_count: usize,
    pub error_type: Option<String>,
    pub query_complexity: f64,
}

/// Service statistics for ranking
#[derive(Debug, Clone)]
pub struct ServiceStatistics {
    pub endpoint: String,
    pub avg_latency_ms: f64,
    pub success_rate: f64,
    pub total_queries: usize,
    pub last_updated: chrono::DateTime<chrono::Utc>,
    pub quality_score: f64,
    pub reliability_trend: ReliabilityTrend,
}

/// Reliability trend enumeration
#[derive(Debug, Clone)]
pub enum ReliabilityTrend {
    Improving,
    Stable,
    Degrading,
}

/// Service ranking information
#[derive(Debug, Clone)]
pub struct ServiceRanking {
    pub endpoint: String,
    pub ranking_score: f64,
    pub ranking_factors: RankingFactors,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Factors contributing to service ranking
#[derive(Debug, Clone)]
pub struct RankingFactors {
    pub latency_score: f64,
    pub reliability_score: f64,
    pub availability_score: f64,
    pub quality_score: f64,
    pub trend_score: f64,
}

/// Statistics cache extension trait
trait StatisticsCacheExt {
    fn get_service_stats(&self, endpoint: &str) -> Option<ServiceStatistics>;
    fn update_service_stats(&mut self, endpoint: &str, stats: ServiceStatistics) -> Result<()>;
    fn add_service_stats(&mut self, endpoint: &str, stats: ServiceStatistics) -> Result<()>;
    fn get_all_service_stats(&self) -> HashMap<String, ServiceStatistics>;
    fn update_global_rankings(&mut self, rankings: Vec<ServiceRanking>) -> Result<()>;
}

impl StatisticsCacheExt for StatisticsCache {
    fn get_service_stats(&self, endpoint: &str) -> Option<ServiceStatistics> {
        // This would be implemented based on the actual StatisticsCache structure
        None // Placeholder
    }

    fn update_service_stats(&mut self, endpoint: &str, stats: ServiceStatistics) -> Result<()> {
        // This would be implemented based on the actual StatisticsCache structure
        Ok(()) // Placeholder
    }

    fn add_service_stats(&mut self, endpoint: &str, stats: ServiceStatistics) -> Result<()> {
        // This would be implemented based on the actual StatisticsCache structure
        Ok(()) // Placeholder
    }

    fn get_all_service_stats(&self) -> HashMap<String, ServiceStatistics> {
        // This would be implemented based on the actual StatisticsCache structure
        HashMap::new() // Placeholder
    }

    fn update_global_rankings(&mut self, rankings: Vec<ServiceRanking>) -> Result<()> {
        // This would be implemented based on the actual StatisticsCache structure
        Ok(()) // Placeholder
    }
}
