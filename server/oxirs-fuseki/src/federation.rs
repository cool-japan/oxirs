//! Advanced SPARQL federation support with intelligent query planning
//!
//! This module provides sophisticated federation capabilities for distributed
//! SPARQL query execution across multiple endpoints.

use crate::{
    error::{FusekiError, FusekiResult},
    store::Store,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn, instrument};

/// Federation query planner with cost-based optimization
#[derive(Clone)]
pub struct FederationPlanner {
    endpoints: Arc<RwLock<HashMap<String, ServiceEndpoint>>>,
    query_cache: Arc<RwLock<HashMap<String, CachedQueryPlan>>>,
    statistics: Arc<RwLock<FederationStatistics>>,
    config: FederationConfig,
}

/// Service endpoint configuration and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    pub url: String,
    pub name: String,
    pub capabilities: EndpointCapabilities,
    pub statistics: EndpointStatistics,
    pub health_status: HealthStatus,
    pub authentication: Option<EndpointAuth>,
    pub timeout_ms: u64,
    pub priority: i32,
}

/// Endpoint capabilities and feature support
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EndpointCapabilities {
    pub sparql_version: String,
    pub supported_features: HashSet<String>,
    pub supported_functions: HashSet<String>,
    pub max_query_length: Option<usize>,
    pub supports_union: bool,
    pub supports_subqueries: bool,
    pub supports_aggregation: bool,
    pub supports_property_paths: bool,
    pub supports_negation: bool,
    pub result_formats: HashSet<String>,
}

/// Endpoint performance statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EndpointStatistics {
    pub average_response_time_ms: f64,
    pub success_rate: f64,
    pub total_queries: u64,
    pub failed_queries: u64,
    pub last_query_time: Option<DateTime<Utc>>,
    pub estimated_cardinality: HashMap<String, u64>,
    pub query_pattern_performance: HashMap<String, f64>,
}

/// Endpoint health status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Authentication configuration for endpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointAuth {
    pub auth_type: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub api_key: Option<String>,
    pub oauth_token: Option<String>,
}

/// Federation configuration
#[derive(Debug, Clone)]
pub struct FederationConfig {
    pub max_concurrent_requests: usize,
    pub default_timeout_ms: u64,
    pub enable_query_optimization: bool,
    pub cache_query_plans: bool,
    pub enable_parallel_execution: bool,
    pub cost_threshold: f64,
}

/// Federated query execution plan
#[derive(Debug, Clone)]
pub struct FederatedQueryPlan {
    pub plan_id: String,
    pub original_query: String,
    pub execution_stages: Vec<ExecutionStage>,
    pub estimated_cost: f64,
    pub estimated_result_count: u64,
    pub parallelizable: bool,
    pub created_at: DateTime<Utc>,
}

/// Query execution stage
#[derive(Debug, Clone)]
pub struct ExecutionStage {
    pub stage_id: String,
    pub stage_type: StageType,
    pub endpoint: String,
    pub query_fragment: String,
    pub dependencies: Vec<String>,
    pub estimated_cost: f64,
    pub estimated_cardinality: u64,
    pub timeout_ms: u64,
}

/// Type of execution stage
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageType {
    Service,        // SERVICE clause execution
    LocalJoin,      // Local join operation
    Union,          // Union of results
    Filter,         // Post-processing filter
    Aggregation,    // Aggregation operation
    Sort,           // Result ordering
    Projection,     // Variable projection
}

/// Cached query plan
#[derive(Debug, Clone)]
pub struct CachedQueryPlan {
    pub plan: FederatedQueryPlan,
    pub cached_at: DateTime<Utc>,
    pub hits: u64,
    pub average_execution_time: f64,
}

/// Federation statistics
#[derive(Debug, Clone, Default)]
pub struct FederationStatistics {
    pub total_federated_queries: u64,
    pub successful_queries: u64,
    pub failed_queries: u64,
    pub average_planning_time_ms: f64,
    pub average_execution_time_ms: f64,
    pub cache_hit_ratio: f64,
    pub parallel_execution_ratio: f64,
}

/// Query execution result
#[derive(Debug, Clone)]
pub struct FederatedQueryResult {
    pub bindings: Vec<HashMap<String, serde_json::Value>>,
    pub execution_plan: FederatedQueryPlan,
    pub execution_time_ms: u64,
    pub endpoint_times: HashMap<String, u64>,
    pub result_count: usize,
}

impl FederationPlanner {
    /// Create new federation planner
    pub fn new(config: FederationConfig) -> Self {
        FederationPlanner {
            endpoints: Arc::new(RwLock::new(HashMap::new())),
            query_cache: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(RwLock::new(FederationStatistics::default())),
            config,
        }
    }

    /// Register a SERVICE endpoint
    pub async fn register_endpoint(&self, endpoint: ServiceEndpoint) -> FusekiResult<()> {
        let mut endpoints = self.endpoints.write().await;
        
        // Test endpoint health before registration
        let health = self.check_endpoint_health(&endpoint).await;
        let mut endpoint = endpoint;
        endpoint.health_status = health;
        
        endpoints.insert(endpoint.url.clone(), endpoint.clone());
        info!("Registered federation endpoint: {} ({})", endpoint.name, endpoint.url);
        
        Ok(())
    }

    /// Remove an endpoint
    pub async fn unregister_endpoint(&self, url: &str) -> FusekiResult<bool> {
        let mut endpoints = self.endpoints.write().await;
        let removed = endpoints.remove(url).is_some();
        
        if removed {
            info!("Unregistered federation endpoint: {}", url);
        }
        
        Ok(removed)
    }

    /// Plan federated query execution
    #[instrument(skip(self, query))]
    pub async fn plan_federated_query(&self, query: &str, local_store: &Store) -> FusekiResult<FederatedQueryPlan> {
        let start_time = Instant::now();
        
        // Check cache first
        if self.config.cache_query_plans {
            if let Some(cached_plan) = self.get_cached_plan(query).await {
                debug!("Using cached query plan for federated query");
                return Ok(cached_plan.plan);
            }
        }

        // Parse query to identify SERVICE clauses
        let service_patterns = self.extract_service_patterns(query)?;
        
        if service_patterns.is_empty() {
            return Err(FusekiError::bad_request("No SERVICE clauses found in query"));
        }

        // Analyze query complexity and estimate costs
        let query_analysis = self.analyze_query_complexity(query).await?;
        
        // Generate execution plan
        let plan = self.generate_execution_plan(query, service_patterns, query_analysis).await?;
        
        // Cache the plan if enabled
        if self.config.cache_query_plans {
            self.cache_query_plan(query, plan.clone()).await;
        }

        let planning_time = start_time.elapsed();
        debug!("Generated federated query plan in {:?}", planning_time);
        
        // Update statistics
        self.update_planning_statistics(planning_time).await;
        
        Ok(plan)
    }

    /// Execute federated query
    #[instrument(skip(self, plan, local_store))]
    pub async fn execute_federated_query(
        &self,
        plan: FederatedQueryPlan,
        local_store: &Store,
    ) -> FusekiResult<FederatedQueryResult> {
        let start_time = Instant::now();
        
        info!("Executing federated query plan: {}", plan.plan_id);
        
        let result = if plan.parallelizable && self.config.enable_parallel_execution {
            self.execute_parallel_plan(plan.clone(), local_store).await?
        } else {
            self.execute_sequential_plan(plan.clone(), local_store).await?
        };
        
        let execution_time = start_time.elapsed();
        
        // Update endpoint statistics
        self.update_endpoint_statistics(&result.endpoint_times).await;
        
        // Update federation statistics
        self.update_execution_statistics(execution_time, true).await;
        
        info!("Federated query executed in {:?}, {} results", execution_time, result.result_count);
        
        Ok(result)
    }

    /// Extract SERVICE patterns from query
    fn extract_service_patterns(&self, query: &str) -> FusekiResult<Vec<ServicePattern>> {
        let mut patterns = Vec::new();
        let query_upper = query.to_uppercase();
        
        // Simple pattern matching for SERVICE clauses
        let lines: Vec<&str> = query.lines().collect();
        let mut in_service = false;
        let mut current_service = None;
        let mut service_content = String::new();
        let mut brace_count = 0;
        
        for line in lines {
            let line_upper = line.to_uppercase();
            
            if line_upper.trim().starts_with("SERVICE") {
                // Extract endpoint URL
                if let Some(start) = line.find('<') {
                    if let Some(end) = line.find('>') {
                        let endpoint = line[start+1..end].to_string();
                        current_service = Some(endpoint);
                        in_service = true;
                        service_content.clear();
                    }
                }
            }
            
            if in_service {
                for ch in line.chars() {
                    match ch {
                        '{' => brace_count += 1,
                        '}' => {
                            brace_count -= 1;
                            if brace_count == 0 {
                                // End of SERVICE block
                                if let Some(endpoint) = current_service.take() {
                                    patterns.push(ServicePattern {
                                        endpoint,
                                        query_pattern: service_content.trim().to_string(),
                                        variables: self.extract_variables(&service_content),
                                        estimated_complexity: self.estimate_pattern_complexity(&service_content),
                                    });
                                }
                                in_service = false;
                                service_content.clear();
                            }
                        }
                        _ => {}
                    }
                }
                
                if in_service && brace_count > 0 {
                    service_content.push_str(line);
                    service_content.push('\n');
                }
            }
        }
        
        if patterns.is_empty() {
            return Err(FusekiError::bad_request("No valid SERVICE patterns found"));
        }
        
        Ok(patterns)
    }

    /// Analyze query complexity
    async fn analyze_query_complexity(&self, query: &str) -> FusekiResult<QueryAnalysis> {
        let complexity_factors = ComplexityFactors {
            service_count: query.to_uppercase().matches("SERVICE").count(),
            join_count: query.matches("JOIN").count() + query.matches(".").count(),
            union_count: query.to_uppercase().matches("UNION").count(),
            optional_count: query.to_uppercase().matches("OPTIONAL").count(),
            filter_count: query.to_uppercase().matches("FILTER").count(),
            aggregation_count: self.count_aggregations(query),
            subquery_count: query.to_uppercase().matches("SELECT").count().saturating_sub(1),
            query_length: query.len(),
        };
        
        let estimated_cost = self.calculate_query_cost(&complexity_factors).await;
        
        Ok(QueryAnalysis {
            complexity_factors,
            estimated_cost,
            parallelizable: self.is_parallelizable(query),
            optimization_hints: self.generate_optimization_hints(query).await,
        })
    }

    /// Generate execution plan
    async fn generate_execution_plan(
        &self,
        query: &str,
        service_patterns: Vec<ServicePattern>,
        analysis: QueryAnalysis,
    ) -> FusekiResult<FederatedQueryPlan> {
        let plan_id = uuid::Uuid::new_v4().to_string();
        let mut execution_stages = Vec::new();
        
        // Create stages for each SERVICE pattern
        for (index, pattern) in service_patterns.iter().enumerate() {
            let stage = ExecutionStage {
                stage_id: format!("service_{}", index),
                stage_type: StageType::Service,
                endpoint: pattern.endpoint.clone(),
                query_fragment: pattern.query_pattern.clone(),
                dependencies: Vec::new(),
                estimated_cost: pattern.estimated_complexity,
                estimated_cardinality: self.estimate_result_cardinality(&pattern.endpoint, &pattern.query_pattern).await,
                timeout_ms: self.get_endpoint_timeout(&pattern.endpoint).await,
            };
            execution_stages.push(stage);
        }
        
        // Add post-processing stages if needed
        if service_patterns.len() > 1 {
            // Add join stage
            execution_stages.push(ExecutionStage {
                stage_id: "join_results".to_string(),
                stage_type: StageType::LocalJoin,
                endpoint: "local".to_string(),
                query_fragment: "JOIN".to_string(),
                dependencies: execution_stages.iter().map(|s| s.stage_id.clone()).collect(),
                estimated_cost: 10.0,
                estimated_cardinality: 1000,
                timeout_ms: 5000,
            });
        }
        
        let plan = FederatedQueryPlan {
            plan_id,
            original_query: query.to_string(),
            execution_stages,
            estimated_cost: analysis.estimated_cost,
            estimated_result_count: 1000, // Simplified estimation
            parallelizable: analysis.parallelizable,
            created_at: Utc::now(),
        };
        
        Ok(plan)
    }

    /// Execute plan in parallel
    async fn execute_parallel_plan(
        &self,
        plan: FederatedQueryPlan,
        local_store: &Store,
    ) -> FusekiResult<FederatedQueryResult> {
        let mut endpoint_times = HashMap::new();
        let mut all_bindings = Vec::new();
        
        // Execute SERVICE stages in parallel
        let service_stages: Vec<_> = plan.execution_stages
            .iter()
            .filter(|stage| matches!(stage.stage_type, StageType::Service))
            .collect();
        
        let mut parallel_tasks = Vec::new();
        
        for stage in service_stages {
            let stage_clone = stage.clone();
            let endpoints = Arc::clone(&self.endpoints);
            
            let task = tokio::spawn(async move {
                let start_time = Instant::now();
                
                // Execute query against endpoint
                let result = Self::execute_service_query(&endpoints, &stage_clone).await;
                
                let execution_time = start_time.elapsed();
                (stage_clone.endpoint.clone(), result, execution_time)
            });
            
            parallel_tasks.push(task);
        }
        
        // Wait for all parallel tasks to complete
        for task in parallel_tasks {
            match task.await {
                Ok((endpoint, Ok(bindings), execution_time)) => {
                    endpoint_times.insert(endpoint, execution_time.as_millis() as u64);
                    all_bindings.extend(bindings);
                }
                Ok((endpoint, Err(e), execution_time)) => {
                    endpoint_times.insert(endpoint.clone(), execution_time.as_millis() as u64);
                    warn!("Service query failed for endpoint {}: {}", endpoint, e);
                }
                Err(e) => {
                    error!("Parallel task failed: {}", e);
                }
            }
        }
        
        // Merge and deduplicate results
        let merged_bindings = self.merge_and_deduplicate_results(all_bindings);
        
        Ok(FederatedQueryResult {
            bindings: merged_bindings.clone(),
            execution_plan: plan,
            execution_time_ms: endpoint_times.values().sum(),
            endpoint_times,
            result_count: merged_bindings.len(),
        })
    }

    /// Execute plan sequentially
    async fn execute_sequential_plan(
        &self,
        plan: FederatedQueryPlan,
        local_store: &Store,
    ) -> FusekiResult<FederatedQueryResult> {
        let mut endpoint_times = HashMap::new();
        let mut all_bindings = Vec::new();
        
        // Execute stages in dependency order
        for stage in &plan.execution_stages {
            if matches!(stage.stage_type, StageType::Service) {
                let start_time = Instant::now();
                
                match Self::execute_service_query(&self.endpoints, stage).await {
                    Ok(bindings) => {
                        all_bindings.extend(bindings);
                    }
                    Err(e) => {
                        warn!("Service query failed for {}: {}", stage.endpoint, e);
                    }
                }
                
                let execution_time = start_time.elapsed();
                endpoint_times.insert(stage.endpoint.clone(), execution_time.as_millis() as u64);
            }
        }
        
        Ok(FederatedQueryResult {
            bindings: all_bindings.clone(),
            execution_plan: plan,
            execution_time_ms: endpoint_times.values().sum(),
            endpoint_times,
            result_count: all_bindings.len(),
        })
    }

    /// Execute query against a SERVICE endpoint
    async fn execute_service_query(
        endpoints: &Arc<RwLock<HashMap<String, ServiceEndpoint>>>,
        stage: &ExecutionStage,
    ) -> FusekiResult<Vec<HashMap<String, serde_json::Value>>> {
        debug!("Executing SERVICE query against: {}", stage.endpoint);
        
        // Get endpoint configuration
        let endpoint = {
            let endpoints_guard = endpoints.read().await;
            endpoints_guard.get(&stage.endpoint)
                .cloned()
                .ok_or_else(|| FusekiError::not_found(format!("Endpoint not found: {}", stage.endpoint)))?
        };
        
        // Create HTTP client with timeout
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(stage.timeout_ms))
            .build()
            .map_err(|e| FusekiError::internal(format!("Failed to create HTTP client: {}", e)))?;
        
        // Prepare query parameters
        let mut params = vec![("query", stage.query_fragment.as_str())];
        
        // Add authentication if configured
        let mut request = client.post(&endpoint.url);
        
        if let Some(auth) = &endpoint.authentication {
            match auth.auth_type.as_str() {
                "basic" => {
                    if let (Some(username), Some(password)) = (&auth.username, &auth.password) {
                        request = request.basic_auth(username, Some(password));
                    }
                }
                "bearer" => {
                    if let Some(token) = &auth.oauth_token {
                        request = request.bearer_auth(token);
                    }
                }
                "api_key" => {
                    if let Some(api_key) = &auth.api_key {
                        request = request.header("X-API-Key", api_key);
                    }
                }
                _ => {}
            }
        }
        
        // Execute query
        let response = request
            .header("Accept", "application/sparql-results+json")
            .form(&params)
            .send()
            .await
            .map_err(|e| FusekiError::service_unavailable(format!("Service request failed: {}", e)))?;
        
        if !response.status().is_success() {
            return Err(FusekiError::service_unavailable(
                format!("Service returned status: {}", response.status())
            ));
        }
        
        // Parse response
        let result: serde_json::Value = response.json().await
            .map_err(|e| FusekiError::parse(format!("Failed to parse service response: {}", e)))?;
        
        // Extract bindings from SPARQL results JSON
        let bindings = result
            .get("results")
            .and_then(|r| r.get("bindings"))
            .and_then(|b| b.as_array())
            .map(|bindings| {
                bindings.iter()
                    .filter_map(|binding| {
                        if let Some(obj) = binding.as_object() {
                            Some(obj.iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect::<HashMap<String, serde_json::Value>>())
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        
        Ok(bindings)
    }

    /// Check endpoint health
    async fn check_endpoint_health(&self, endpoint: &ServiceEndpoint) -> HealthStatus {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(endpoint.timeout_ms))
            .build();
        
        let client = match client {
            Ok(c) => c,
            Err(_) => return HealthStatus::Unknown,
        };
        
        // Simple health check with basic SPARQL query
        let health_query = "ASK { ?s ?p ?o }";
        let params = vec![("query", health_query)];
        
        match client.post(&endpoint.url).form(&params).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    HealthStatus::Healthy
                } else {
                    HealthStatus::Degraded
                }
            }
            Err(_) => HealthStatus::Unhealthy,
        }
    }

    /// Helper methods for query analysis
    fn extract_variables(&self, query_pattern: &str) -> Vec<String> {
        let mut variables = Vec::new();
        
        for token in query_pattern.split_whitespace() {
            if token.starts_with('?') || token.starts_with('$') {
                variables.push(token.to_string());
            }
        }
        
        variables.sort();
        variables.dedup();
        variables
    }

    fn estimate_pattern_complexity(&self, pattern: &str) -> f64 {
        let factors = [
            (pattern.matches("?").count() as f64) * 2.0,    // Variables
            (pattern.matches(".").count() as f64) * 1.0,    // Triple patterns
            (pattern.matches("OPTIONAL").count() as f64) * 5.0,  // Optional patterns
            (pattern.matches("UNION").count() as f64) * 3.0,     // Union patterns
            (pattern.matches("FILTER").count() as f64) * 2.0,    // Filters
        ];
        
        factors.iter().sum::<f64>().max(1.0)
    }

    fn count_aggregations(&self, query: &str) -> usize {
        let query_upper = query.to_uppercase();
        ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX(", "GROUP_CONCAT(", "SAMPLE("]
            .iter()
            .map(|agg| query_upper.matches(agg).count())
            .sum()
    }

    async fn calculate_query_cost(&self, factors: &ComplexityFactors) -> f64 {
        let base_cost = 10.0;
        let service_cost = factors.service_count as f64 * 20.0;
        let join_cost = factors.join_count as f64 * 5.0;
        let complexity_cost = (factors.query_length as f64 / 100.0) * 2.0;
        
        base_cost + service_cost + join_cost + complexity_cost
    }

    fn is_parallelizable(&self, query: &str) -> bool {
        // Simple heuristic: queries with multiple independent SERVICE clauses can be parallelized
        let service_count = query.to_uppercase().matches("SERVICE").count();
        service_count > 1 && !query.to_uppercase().contains("OPTIONAL")
    }

    async fn generate_optimization_hints(&self, query: &str) -> Vec<String> {
        let mut hints = Vec::new();
        
        if query.to_uppercase().matches("SERVICE").count() > 1 {
            hints.push("Consider parallel execution".to_string());
        }
        
        if query.matches("FILTER").count() > 2 {
            hints.push("Filter pushdown recommended".to_string());
        }
        
        if query.len() > 1000 {
            hints.push("Query complexity is high".to_string());
        }
        
        hints
    }

    async fn estimate_result_cardinality(&self, endpoint: &str, pattern: &str) -> u64 {
        // Simplified cardinality estimation
        let endpoints = self.endpoints.read().await;
        if let Some(endpoint_info) = endpoints.get(endpoint) {
            endpoint_info.statistics.estimated_cardinality
                .get("default")
                .copied()
                .unwrap_or(100)
        } else {
            100
        }
    }

    async fn get_endpoint_timeout(&self, endpoint: &str) -> u64 {
        let endpoints = self.endpoints.read().await;
        if let Some(endpoint_info) = endpoints.get(endpoint) {
            endpoint_info.timeout_ms
        } else {
            self.config.default_timeout_ms
        }
    }

    fn merge_and_deduplicate_results(
        &self,
        bindings: Vec<HashMap<String, serde_json::Value>>,
    ) -> Vec<HashMap<String, serde_json::Value>> {
        let mut seen = HashSet::new();
        let mut unique_bindings = Vec::new();
        
        for binding in bindings {
            let binding_str = serde_json::to_string(&binding).unwrap_or_default();
            if seen.insert(binding_str) {
                unique_bindings.push(binding);
            }
        }
        
        unique_bindings
    }

    // Cache management methods
    async fn get_cached_plan(&self, query: &str) -> Option<CachedQueryPlan> {
        let cache = self.query_cache.read().await;
        cache.get(query).cloned()
    }

    async fn cache_query_plan(&self, query: &str, plan: FederatedQueryPlan) {
        let mut cache = self.query_cache.write().await;
        let cached_plan = CachedQueryPlan {
            plan,
            cached_at: Utc::now(),
            hits: 0,
            average_execution_time: 0.0,
        };
        cache.insert(query.to_string(), cached_plan);
    }

    // Statistics update methods
    async fn update_planning_statistics(&self, planning_time: Duration) {
        let mut stats = self.statistics.write().await;
        stats.average_planning_time_ms = 
            (stats.average_planning_time_ms + planning_time.as_millis() as f64) / 2.0;
    }

    async fn update_execution_statistics(&self, execution_time: Duration, success: bool) {
        let mut stats = self.statistics.write().await;
        stats.total_federated_queries += 1;
        
        if success {
            stats.successful_queries += 1;
        } else {
            stats.failed_queries += 1;
        }
        
        stats.average_execution_time_ms = 
            (stats.average_execution_time_ms + execution_time.as_millis() as f64) / 2.0;
    }

    async fn update_endpoint_statistics(&self, endpoint_times: &HashMap<String, u64>) {
        let mut endpoints = self.endpoints.write().await;
        
        for (endpoint_url, time_ms) in endpoint_times {
            if let Some(endpoint) = endpoints.get_mut(endpoint_url) {
                let current_avg = endpoint.statistics.average_response_time_ms;
                endpoint.statistics.average_response_time_ms = 
                    (current_avg + *time_ms as f64) / 2.0;
                endpoint.statistics.total_queries += 1;
                endpoint.statistics.last_query_time = Some(Utc::now());
            }
        }
    }

    /// Get federation statistics
    pub async fn get_federation_statistics(&self) -> FederationStatistics {
        let stats = self.statistics.read().await;
        stats.clone()
    }

    /// Get endpoint information
    pub async fn get_endpoints(&self) -> HashMap<String, ServiceEndpoint> {
        let endpoints = self.endpoints.read().await;
        endpoints.clone()
    }
}

/// Supporting data structures

#[derive(Debug, Clone)]
pub struct ServicePattern {
    pub endpoint: String,
    pub query_pattern: String,
    pub variables: Vec<String>,
    pub estimated_complexity: f64,
}

#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    pub complexity_factors: ComplexityFactors,
    pub estimated_cost: f64,
    pub parallelizable: bool,
    pub optimization_hints: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ComplexityFactors {
    pub service_count: usize,
    pub join_count: usize,
    pub union_count: usize,
    pub optional_count: usize,
    pub filter_count: usize,
    pub aggregation_count: usize,
    pub subquery_count: usize,
    pub query_length: usize,
}

impl Default for FederationConfig {
    fn default() -> Self {
        FederationConfig {
            max_concurrent_requests: 10,
            default_timeout_ms: 30000,
            enable_query_optimization: true,
            cache_query_plans: true,
            enable_parallel_execution: true,
            cost_threshold: 100.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_federation_planner_creation() {
        let config = FederationConfig::default();
        let planner = FederationPlanner::new(config);
        
        let endpoints = planner.get_endpoints().await;
        assert!(endpoints.is_empty());
    }

    #[tokio::test]
    async fn test_endpoint_registration() {
        let config = FederationConfig::default();
        let planner = FederationPlanner::new(config);
        
        let endpoint = ServiceEndpoint {
            url: "http://example.org/sparql".to_string(),
            name: "Example Endpoint".to_string(),
            capabilities: EndpointCapabilities::default(),
            statistics: EndpointStatistics::default(),
            health_status: HealthStatus::Unknown,
            authentication: None,
            timeout_ms: 30000,
            priority: 1,
        };
        
        planner.register_endpoint(endpoint).await.unwrap();
        
        let endpoints = planner.get_endpoints().await;
        assert_eq!(endpoints.len(), 1);
        assert!(endpoints.contains_key("http://example.org/sparql"));
    }

    #[test]
    fn test_service_pattern_extraction() {
        let planner = FederationPlanner::new(FederationConfig::default());
        
        let query = r#"
            SELECT * WHERE {
                SERVICE <http://example.org/sparql> {
                    ?s ?p ?o
                }
                SERVICE <http://other.org/sparql> {
                    ?s ?p2 ?o2
                }
            }
        "#;
        
        let patterns = planner.extract_service_patterns(query).unwrap();
        assert_eq!(patterns.len(), 2);
        assert_eq!(patterns[0].endpoint, "http://example.org/sparql");
        assert_eq!(patterns[1].endpoint, "http://other.org/sparql");
    }

    #[test]
    fn test_complexity_estimation() {
        let planner = FederationPlanner::new(FederationConfig::default());
        
        let simple_pattern = "?s ?p ?o";
        let complex_pattern = "?s ?p ?o . OPTIONAL { ?s ?p2 ?o2 } FILTER(?o > 10)";
        
        let simple_complexity = planner.estimate_pattern_complexity(simple_pattern);
        let complex_complexity = planner.estimate_pattern_complexity(complex_pattern);
        
        assert!(complex_complexity > simple_complexity);
    }
}