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
    pub execution_steps: Vec<ExecutionStep>,
    pub estimated_cost: f64,
    pub parallel_sections: Vec<ParallelSection>,
    pub data_flow: DataFlowGraph,
    pub resource_requirements: ResourceRequirements,
}

/// Individual execution step in federated plan
#[derive(Debug, Clone)]
pub struct ExecutionStep {
    pub step_id: String,
    pub step_type: ExecutionStepType,
    pub target_endpoint: Option<String>,
    pub query_fragment: String,
    pub dependencies: Vec<String>,
    pub estimated_execution_time: Duration,
    pub estimated_result_size: usize,
}

/// Types of execution steps
#[derive(Debug, Clone)]
pub enum ExecutionStepType {
    LocalQuery,
    RemoteServiceCall,
    Join,
    Union,
    Filter,
    Project,
    Sort,
    Limit,
    Aggregation,
}

/// Parallel execution section
#[derive(Debug, Clone)]
pub struct ParallelSection {
    pub section_id: String,
    pub parallel_steps: Vec<String>, // Step IDs that can run in parallel
    pub synchronization_point: String,
    pub max_parallelism: usize,
}

/// Data flow graph for query execution
#[derive(Debug, Clone)]
pub struct DataFlowGraph {
    pub nodes: HashMap<String, DataFlowNode>,
    pub edges: Vec<DataFlowEdge>,
}

#[derive(Debug, Clone)]
pub struct DataFlowNode {
    pub node_id: String,
    pub operation: String,
    pub estimated_cardinality: u64,
    pub estimated_selectivity: f64,
}

#[derive(Debug, Clone)]
pub struct DataFlowEdge {
    pub from_node: String,
    pub to_node: String,
    pub data_transfer_cost: f64,
}

/// Resource requirements for query execution
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub estimated_memory_mb: f64,
    pub estimated_cpu_seconds: f64,
    pub estimated_network_mb: f64,
    pub required_endpoints: HashSet<String>,
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
    pub average_execution_time: f64,
    pub endpoint_performance: HashMap<String, EndpointPerformanceMetrics>,
    pub query_distribution: HashMap<String, u64>,
    pub failure_rates: HashMap<String, f64>,
}

/// Performance metrics per endpoint
#[derive(Debug, Clone, Default)]
pub struct EndpointPerformanceMetrics {
    pub average_response_time: f64,
    pub success_rate: f64,
    pub throughput_queries_per_second: f64,
    pub current_load: f64,
}

impl FederationPlanner {
    /// Create new federation planner
    pub fn new() -> Self {
        Self {
            endpoints: Arc::new(RwLock::new(HashMap::new())),
            query_cache: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(RwLock::new(FederationStatistics::default())),
            config: FederationConfig::default(),
        }
    }
    
    /// Create execution plan for federated query
    pub async fn create_execution_plan(&self, query: &str) -> FusekiResult<FederatedQueryPlan> {
        let plan_id = uuid::Uuid::new_v4().to_string();
        
        // Check cache first
        let cache_key = self.generate_cache_key(query);
        if let Ok(cache) = self.query_cache.read().await {
            if let Some(cached_plan) = cache.get(&cache_key) {
                info!("Using cached federated query plan: {}", plan_id);
                return Ok(cached_plan.plan.clone());
            }
        }
        
        // Parse query and identify SERVICE clauses
        let service_clauses = self.extract_service_clauses(query)?;
        
        // Analyze query complexity and data requirements
        let query_analysis = self.analyze_query_complexity(query).await?;
        
        // Create execution steps
        let execution_steps = self.create_execution_steps(query, &service_clauses).await?;
        
        // Determine parallel execution opportunities
        let parallel_sections = self.identify_parallel_sections(&execution_steps)?;
        
        // Build data flow graph
        let data_flow = self.build_data_flow_graph(&execution_steps)?;
        
        // Estimate resource requirements
        let resource_requirements = self.estimate_resource_requirements(&execution_steps, &query_analysis).await?;
        
        let plan = FederatedQueryPlan {
            plan_id: plan_id.clone(),
            original_query: query.to_string(),
            execution_steps,
            estimated_cost: query_analysis.estimated_cost,
            parallel_sections,
            data_flow,
            resource_requirements,
        };
        
        // Cache the plan if enabled
        if self.config.cache_query_plans {
            if let Ok(mut cache) = self.query_cache.write().await {
                cache.insert(cache_key, CachedQueryPlan {
                    plan: plan.clone(),
                    cached_at: Utc::now(),
                    hits: 0,
                    average_execution_time: 0.0,
                });
            }
        }
        
        info!("Created federated query plan: {} with {} steps", 
              plan_id, plan.execution_steps.len());
        
        Ok(plan)
    }
    
    /// Add endpoint to federation
    pub async fn add_endpoint(&self, endpoint: ServiceEndpoint) -> FusekiResult<()> {
        let mut endpoints = self.endpoints.write().await;
        endpoints.insert(endpoint.url.clone(), endpoint);
        Ok(())
    }
    
    /// Remove endpoint from federation
    pub async fn remove_endpoint(&self, url: &str) -> FusekiResult<bool> {
        let mut endpoints = self.endpoints.write().await;
        Ok(endpoints.remove(url).is_some())
    }
    
    /// Get endpoint information
    pub async fn get_endpoint(&self, url: &str) -> Option<ServiceEndpoint> {
        let endpoints = self.endpoints.read().await;
        endpoints.get(url).cloned()
    }
    
    /// Update endpoint health status
    pub async fn update_endpoint_health(&self, url: &str, status: HealthStatus) -> FusekiResult<()> {
        let mut endpoints = self.endpoints.write().await;
        if let Some(endpoint) = endpoints.get_mut(url) {
            endpoint.health_status = status;
        }
        Ok(())
    }
    
    /// Get federation statistics
    pub async fn get_statistics(&self) -> FederationStatistics {
        let stats = self.statistics.read().await;
        stats.clone()
    }
    
    // Helper methods
    
    fn generate_cache_key(&self, query: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        format!("federated_plan_{:x}", hasher.finish())
    }
    
    fn extract_service_clauses(&self, query: &str) -> FusekiResult<Vec<ServiceClause>> {
        let mut service_clauses = Vec::new();
        
        // Simple regex-based extraction (in production, use proper SPARQL parser)
        if let Ok(service_regex) = regex::Regex::new(r"SERVICE\s+<([^>]+)>\s*\{([^}]+)\}") {
            for captures in service_regex.captures_iter(query) {
                if let (Some(endpoint), Some(query_fragment)) = 
                   (captures.get(1), captures.get(2)) {
                    service_clauses.push(ServiceClause {
                        endpoint_url: endpoint.as_str().to_string(),
                        query_fragment: query_fragment.as_str().to_string(),
                        variables: self.extract_variables(query_fragment.as_str()),
                        optional: query.contains("OPTIONAL"),
                    });
                }
            }
        }
        
        Ok(service_clauses)
    }
    
    fn extract_variables(&self, query_fragment: &str) -> Vec<String> {
        let mut variables = Vec::new();
        if let Ok(var_regex) = regex::Regex::new(r"\?(\w+)") {
            for captures in var_regex.captures_iter(query_fragment) {
                if let Some(var) = captures.get(1) {
                    variables.push(var.as_str().to_string());
                }
            }
        }
        variables
    }
    
    async fn analyze_query_complexity(&self, query: &str) -> FusekiResult<QueryAnalysis> {
        // Analyze query complexity based on various factors
        let triple_patterns = query.matches("?").count();
        let has_optional = query.contains("OPTIONAL");
        let has_union = query.contains("UNION");
        let has_filter = query.contains("FILTER");
        let has_order_by = query.contains("ORDER BY");
        let has_group_by = query.contains("GROUP BY");
        
        let complexity_score = (triple_patterns as f64) * 1.0 +
                              (if has_optional { 2.0 } else { 0.0 }) +
                              (if has_union { 1.5 } else { 0.0 }) +
                              (if has_filter { 1.0 } else { 0.0 }) +
                              (if has_order_by { 1.0 } else { 0.0 }) +
                              (if has_group_by { 2.0 } else { 0.0 });
        
        Ok(QueryAnalysis {
            complexity_score,
            estimated_cost: complexity_score * 100.0,
            triple_pattern_count: triple_patterns,
            has_aggregation: has_group_by,
            has_sorting: has_order_by,
            estimated_cardinality: 1000, // Default estimate
        })
    }
    
    async fn create_execution_steps(
        &self, 
        query: &str, 
        service_clauses: &[ServiceClause]
    ) -> FusekiResult<Vec<ExecutionStep>> {
        let mut steps = Vec::new();
        
        // Create steps for SERVICE clauses
        for (i, service) in service_clauses.iter().enumerate() {
            steps.push(ExecutionStep {
                step_id: format!("service_{}", i),
                step_type: ExecutionStepType::RemoteServiceCall,
                target_endpoint: Some(service.endpoint_url.clone()),
                query_fragment: service.query_fragment.clone(),
                dependencies: Vec::new(),
                estimated_execution_time: Duration::from_millis(500),
                estimated_result_size: 100,
            });
        }
        
        // Add local processing steps if needed
        if service_clauses.len() > 1 {
            steps.push(ExecutionStep {
                step_id: "join_results".to_string(),
                step_type: ExecutionStepType::Join,
                target_endpoint: None,
                query_fragment: "JOIN results".to_string(),
                dependencies: (0..service_clauses.len()).map(|i| format!("service_{}", i)).collect(),
                estimated_execution_time: Duration::from_millis(100),
                estimated_result_size: 500,
            });
        }
        
        Ok(steps)
    }
    
    fn identify_parallel_sections(&self, steps: &[ExecutionStep]) -> FusekiResult<Vec<ParallelSection>> {
        let mut parallel_sections = Vec::new();
        
        // Find steps that can run in parallel (no dependencies)
        let independent_steps: Vec<String> = steps.iter()
            .filter(|step| step.dependencies.is_empty())
            .map(|step| step.step_id.clone())
            .collect();
        
        if independent_steps.len() > 1 {
            parallel_sections.push(ParallelSection {
                section_id: "parallel_services".to_string(),
                parallel_steps: independent_steps,
                synchronization_point: "join_results".to_string(),
                max_parallelism: self.config.max_concurrent_requests,
            });
        }
        
        Ok(parallel_sections)
    }
    
    fn build_data_flow_graph(&self, steps: &[ExecutionStep]) -> FusekiResult<DataFlowGraph> {
        let mut nodes = HashMap::new();
        let mut edges = Vec::new();
        
        // Create nodes for each step
        for step in steps {
            nodes.insert(step.step_id.clone(), DataFlowNode {
                node_id: step.step_id.clone(),
                operation: format!("{:?}", step.step_type),
                estimated_cardinality: step.estimated_result_size as u64,
                estimated_selectivity: 0.5, // Default selectivity
            });
        }
        
        // Create edges based on dependencies
        for step in steps {
            for dependency in &step.dependencies {
                edges.push(DataFlowEdge {
                    from_node: dependency.clone(),
                    to_node: step.step_id.clone(),
                    data_transfer_cost: 1.0, // Default cost
                });
            }
        }
        
        Ok(DataFlowGraph { nodes, edges })
    }
    
    async fn estimate_resource_requirements(
        &self, 
        steps: &[ExecutionStep], 
        analysis: &QueryAnalysis
    ) -> FusekiResult<ResourceRequirements> {
        let estimated_memory = steps.iter()
            .map(|step| step.estimated_result_size as f64 * 0.001) // 1KB per result
            .sum::<f64>();
        
        let estimated_cpu = analysis.complexity_score * 0.1; // seconds
        
        let estimated_network = steps.iter()
            .filter(|step| step.target_endpoint.is_some())
            .map(|step| step.estimated_result_size as f64 * 0.002) // 2KB per remote result
            .sum::<f64>();
        
        let required_endpoints = steps.iter()
            .filter_map(|step| step.target_endpoint.as_ref())
            .cloned()
            .collect();
        
        Ok(ResourceRequirements {
            estimated_memory_mb: estimated_memory,
            estimated_cpu_seconds: estimated_cpu,
            estimated_network_mb: estimated_network,
            required_endpoints,
        })
    }
}

impl FederationConfig {
    pub fn default() -> Self {
        Self {
            max_concurrent_requests: 10,
            default_timeout_ms: 30000,
            enable_query_optimization: true,
            cache_query_plans: true,
            enable_parallel_execution: true,
            cost_threshold: 1000.0,
        }
    }
}

/// Service clause extracted from query
#[derive(Debug, Clone)]
pub struct ServiceClause {
    pub endpoint_url: String,
    pub query_fragment: String,
    pub variables: Vec<String>,
    pub optional: bool,
}

/// Query analysis results
#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    pub complexity_score: f64,
    pub estimated_cost: f64,
    pub triple_pattern_count: usize,
    pub has_aggregation: bool,
    pub has_sorting: bool,
    pub estimated_cardinality: u64,
}
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