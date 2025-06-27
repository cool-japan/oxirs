//! Query planning for federated SPARQL execution

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};
use tokio::sync::RwLock;
use url::Url;

use oxirs_core::{
    NamedNode,
    sparql::{Query, QueryType},
};

use crate::{
    error::{Error, Result},
    federation::{FederationConfig, ServiceEndpoint, ServiceHealth},
};

/// Query plan for federated execution
#[derive(Debug, Clone)]
pub struct FederatedQueryPlan {
    /// Original query
    pub query: Query,
    /// Execution steps
    pub steps: Vec<ExecutionStep>,
    /// Estimated cost
    pub estimated_cost: QueryCost,
    /// Preferred execution strategy
    pub strategy: ExecutionStrategy,
}

/// A single step in query execution
#[derive(Debug, Clone)]
pub struct ExecutionStep {
    /// Step identifier
    pub id: String,
    /// Service endpoint(s) for this step
    pub services: Vec<ServiceSelection>,
    /// Sub-query for this step
    pub sub_query: Query,
    /// Dependencies on other steps
    pub dependencies: Vec<String>,
    /// Estimated cost for this step
    pub cost: QueryCost,
}

/// Service selection for a query step
#[derive(Debug, Clone)]
pub struct ServiceSelection {
    /// Service identifier
    pub service_id: String,
    /// Service URL
    pub service_url: Url,
    /// Selection score (higher is better)
    pub score: f64,
    /// Is this the primary choice
    pub is_primary: bool,
}

/// Query cost estimation
#[derive(Debug, Clone, Default)]
pub struct QueryCost {
    /// Estimated result size
    pub result_size: Option<usize>,
    /// Estimated execution time
    pub execution_time: Option<Duration>,
    /// Network transfer cost
    pub network_cost: Option<f64>,
    /// Computational complexity
    pub complexity: Option<f64>,
}

/// Execution strategy for federated queries
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionStrategy {
    /// Execute sequentially
    Sequential,
    /// Execute in parallel
    Parallel,
    /// Adaptive based on runtime conditions
    Adaptive,
}

/// Query planner for federated execution
pub struct QueryPlanner {
    config: FederationConfig,
    endpoints: Arc<RwLock<HashMap<String, ServiceEndpoint>>>,
    statistics: Arc<RwLock<QueryStatistics>>,
}

/// Query execution statistics for cost estimation
#[derive(Debug, Default)]
struct QueryStatistics {
    /// Historical query patterns
    pattern_stats: HashMap<String, PatternStatistics>,
    /// Service performance metrics
    service_stats: HashMap<String, ServiceStatistics>,
}

#[derive(Debug, Default)]
struct PatternStatistics {
    /// Average result size
    avg_result_size: usize,
    /// Average execution time
    avg_execution_time: Duration,
    /// Number of executions
    execution_count: u64,
}

#[derive(Debug, Default)]
struct ServiceStatistics {
    /// Average response time
    avg_response_time: Duration,
    /// Success rate (0.0 - 1.0)
    success_rate: f64,
    /// Average result size
    avg_result_size: usize,
}

impl QueryPlanner {
    /// Create a new query planner
    pub fn new(
        config: FederationConfig,
        endpoints: Arc<RwLock<HashMap<String, ServiceEndpoint>>>,
    ) -> Self {
        Self {
            config,
            endpoints,
            statistics: Arc::new(RwLock::new(QueryStatistics::default())),
        }
    }

    /// Plan a federated query execution
    pub async fn plan_query(&self, query: Query) -> Result<FederatedQueryPlan> {
        // Analyze query structure
        let service_patterns = self.extract_service_patterns(&query)?;
        
        if service_patterns.is_empty() {
            // No SERVICE clauses, execute locally
            return Ok(FederatedQueryPlan {
                query: query.clone(),
                steps: vec![],
                estimated_cost: QueryCost::default(),
                strategy: ExecutionStrategy::Sequential,
            });
        }

        // Get available endpoints
        let endpoints = self.endpoints.read().await;
        let healthy_endpoints: Vec<_> = endpoints
            .iter()
            .filter(|(_, ep)| matches!(ep.health, ServiceHealth::Healthy | ServiceHealth::Degraded))
            .collect();

        if healthy_endpoints.is_empty() {
            return Err(Error::Custom("No healthy service endpoints available".to_string()));
        }

        // Build execution steps
        let mut steps = Vec::new();
        let mut total_cost = QueryCost::default();

        for (i, pattern) in service_patterns.iter().enumerate() {
            let step = self.plan_service_step(
                &format!("step_{}", i),
                pattern,
                &healthy_endpoints,
            ).await?;
            
            // Update total cost
            if let Some(cost) = &step.cost.execution_time {
                total_cost.execution_time = Some(
                    total_cost.execution_time.unwrap_or(Duration::ZERO) + *cost
                );
            }
            
            steps.push(step);
        }

        // Determine execution strategy
        let strategy = self.determine_strategy(&steps);

        Ok(FederatedQueryPlan {
            query,
            steps,
            estimated_cost: total_cost,
            strategy,
        })
    }

    /// Extract SERVICE patterns from query
    fn extract_service_patterns(&self, query: &Query) -> Result<Vec<ServicePattern>> {
        // TODO: Implement proper SPARQL parsing to extract SERVICE clauses
        // For now, return empty vector
        Ok(vec![])
    }

    /// Plan execution for a single service pattern
    async fn plan_service_step(
        &self,
        step_id: &str,
        pattern: &ServicePattern,
        endpoints: &[(&String, &ServiceEndpoint)],
    ) -> Result<ExecutionStep> {
        let mut service_selections = Vec::new();

        // Score each endpoint for this pattern
        for (id, endpoint) in endpoints {
            let score = self.score_endpoint_for_pattern(pattern, endpoint).await;
            
            service_selections.push(ServiceSelection {
                service_id: id.to_string(),
                service_url: endpoint.url.clone(),
                score,
                is_primary: false,
            });
        }

        // Sort by score and mark the best as primary
        service_selections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        if let Some(first) = service_selections.first_mut() {
            first.is_primary = true;
        }

        // Estimate cost
        let cost = self.estimate_pattern_cost(pattern, &service_selections).await;

        Ok(ExecutionStep {
            id: step_id.to_string(),
            services: service_selections,
            sub_query: pattern.query.clone(),
            dependencies: pattern.dependencies.clone(),
            cost,
        })
    }

    /// Score an endpoint for a specific pattern
    async fn score_endpoint_for_pattern(
        &self,
        pattern: &ServicePattern,
        endpoint: &ServiceEndpoint,
    ) -> f64 {
        let mut score = 100.0;

        // Adjust based on health status
        match endpoint.health {
            ServiceHealth::Healthy => score *= 1.0,
            ServiceHealth::Degraded => score *= 0.7,
            ServiceHealth::Unhealthy => return 0.0,
            ServiceHealth::Unknown => score *= 0.5,
        }

        // Adjust based on response time
        if let Some(avg_time) = &endpoint.capabilities.avg_response_time {
            let time_penalty = avg_time.as_secs_f64() / 10.0; // Penalty for slow response
            score *= (1.0 - time_penalty.min(0.9));
        }

        // Adjust based on capabilities
        if pattern.required_features.iter().all(|f| 
            endpoint.capabilities.sparql_features.contains(f)
        ) {
            score *= 1.2; // Bonus for supporting all required features
        }

        // Use statistics if available
        let stats = self.statistics.read().await;
        if let Some(service_stats) = stats.service_stats.get(&endpoint.url.to_string()) {
            score *= service_stats.success_rate;
        }

        score
    }

    /// Estimate cost for a pattern execution
    async fn estimate_pattern_cost(
        &self,
        pattern: &ServicePattern,
        services: &[ServiceSelection],
    ) -> QueryCost {
        let mut cost = QueryCost::default();

        // Use primary service for estimation
        if let Some(primary) = services.iter().find(|s| s.is_primary) {
            let endpoints = self.endpoints.read().await;
            if let Some((_, endpoint)) = endpoints.iter().find(|(id, _)| **id == primary.service_id) {
                // Use endpoint capabilities for estimation
                if let Some(avg_time) = &endpoint.capabilities.avg_response_time {
                    cost.execution_time = Some(*avg_time);
                }
                
                // Estimate based on pattern complexity
                cost.complexity = Some(self.estimate_pattern_complexity(pattern));
            }
        }

        // Use historical statistics
        let stats = self.statistics.read().await;
        let pattern_key = self.pattern_to_key(pattern);
        if let Some(pattern_stats) = stats.pattern_stats.get(&pattern_key) {
            cost.result_size = Some(pattern_stats.avg_result_size);
            if cost.execution_time.is_none() {
                cost.execution_time = Some(pattern_stats.avg_execution_time);
            }
        }

        cost
    }

    /// Estimate pattern complexity
    fn estimate_pattern_complexity(&self, pattern: &ServicePattern) -> f64 {
        // Simple heuristic based on pattern size
        let triple_count = pattern.triple_patterns.len() as f64;
        let filter_count = pattern.filters.len() as f64;
        
        triple_count * 1.0 + filter_count * 0.5
    }

    /// Convert pattern to cache key
    fn pattern_to_key(&self, pattern: &ServicePattern) -> String {
        // Simple key generation - in practice would be more sophisticated
        format!("pattern_{}_triples_{}_filters", 
            pattern.triple_patterns.len(),
            pattern.filters.len()
        )
    }

    /// Determine execution strategy based on steps
    fn determine_strategy(&self, steps: &[ExecutionStep]) -> ExecutionStrategy {
        // Check for dependencies
        let has_dependencies = steps.iter().any(|s| !s.dependencies.is_empty());
        
        if has_dependencies {
            ExecutionStrategy::Sequential
        } else if steps.len() > 1 && self.config.max_concurrent_requests > 1 {
            ExecutionStrategy::Parallel
        } else {
            ExecutionStrategy::Adaptive
        }
    }

    /// Update statistics after query execution
    pub async fn update_statistics(
        &self,
        service_id: &str,
        pattern_key: String,
        result_size: usize,
        execution_time: Duration,
        success: bool,
    ) {
        let mut stats = self.statistics.write().await;
        
        // Update pattern statistics
        let pattern_stats = stats.pattern_stats.entry(pattern_key).or_default();
        pattern_stats.execution_count += 1;
        
        // Moving average update
        let count = pattern_stats.execution_count as f64;
        pattern_stats.avg_result_size = 
            ((pattern_stats.avg_result_size as f64 * (count - 1.0) + result_size as f64) / count) as usize;
        
        let avg_millis = pattern_stats.avg_execution_time.as_millis() as f64;
        let new_millis = execution_time.as_millis() as f64;
        let updated_millis = (avg_millis * (count - 1.0) + new_millis) / count;
        pattern_stats.avg_execution_time = Duration::from_millis(updated_millis as u64);
        
        // Update service statistics
        if let Some(endpoint) = self.endpoints.read().await.get(service_id) {
            let service_stats = stats.service_stats.entry(endpoint.url.to_string()).or_default();
            
            // Update success rate
            let total = service_stats.success_rate * 100.0 + if success { 1.0 } else { 0.0 };
            service_stats.success_rate = total / 101.0;
            
            // Update response time
            service_stats.avg_response_time = 
                (service_stats.avg_response_time + execution_time) / 2;
            
            // Update result size
            service_stats.avg_result_size = 
                (service_stats.avg_result_size + result_size) / 2;
        }
    }
}

/// Service pattern extracted from query
#[derive(Debug, Clone)]
struct ServicePattern {
    /// Service URI
    service_uri: Option<NamedNode>,
    /// Sub-query for this service
    query: Query,
    /// Triple patterns in this service block
    triple_patterns: Vec<String>,
    /// Filters in this service block
    filters: Vec<String>,
    /// Required SPARQL features
    required_features: Vec<String>,
    /// Dependencies on other patterns
    dependencies: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_query_planner_creation() {
        let config = FederationConfig::default();
        let endpoints = Arc::new(RwLock::new(HashMap::new()));
        let planner = QueryPlanner::new(config, endpoints);
        
        // Create a simple query
        let query = Query {
            prefixes: HashMap::new(),
            dataset: None,
            query_type: QueryType::Select {
                distinct: false,
                reduced: false,
                variables: vec![],
                where_clause: vec![],
                group_by: None,
                having: None,
                order_by: None,
                limit: None,
                offset: None,
                values: None,
            },
        };
        
        let plan = planner.plan_query(query).await.unwrap();
        assert_eq!(plan.steps.len(), 0); // No SERVICE clauses
        assert_eq!(plan.strategy, ExecutionStrategy::Sequential);
    }
}