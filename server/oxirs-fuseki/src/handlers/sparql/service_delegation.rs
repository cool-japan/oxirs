//! SERVICE delegation and federation support for SPARQL
//!
//! This module handles SERVICE clauses in SPARQL queries, including
//! endpoint discovery, parallel execution, and result merging.

use crate::error::FusekiResult;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, warn};

/// SERVICE delegation manager
#[derive(Debug, Clone)]
pub struct ServiceDelegationManager {
    endpoints: Arc<RwLock<HashMap<String, ServiceEndpoint>>>,
    executor: ParallelServiceExecutor,
    merger: ServiceResultMerger,
    discovery: EndpointDiscovery,
    health_monitor: HealthMonitor,
    query_cache: Arc<RwLock<QueryCache>>,
    load_balancer: LoadBalancer,
}

/// Service endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    pub url: String,
    pub name: String,
    pub supported_features: HashSet<String>,
    pub authentication: Option<ServiceAuthentication>,
    pub timeout: Duration,
    pub retry_count: u32,
    pub health_status: ServiceHealth,
    pub response_time_avg: Option<Duration>,
    pub last_checked: Option<std::time::SystemTime>,
}

/// Service authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceAuthentication {
    pub auth_type: AuthenticationType,
    pub credentials: HashMap<String, String>,
}

/// Authentication types for services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthenticationType {
    None,
    Basic,
    Bearer,
    ApiKey,
    OAuth2,
    Custom(String),
}

/// Service health status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ServiceHealth {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
    Maintenance,
}

/// Parallel service executor
#[derive(Debug, Clone)]
pub struct ParallelServiceExecutor {
    max_concurrent: usize,
    timeout: Duration,
    retry_policy: RetryPolicy,
}

/// Retry policy for service calls
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
}

/// Service result merger
#[derive(Debug, Clone)]
pub struct ServiceResultMerger {
    merge_strategies: HashMap<String, MergeStrategy>,
    default_strategy: MergeStrategy,
}

/// Merge strategies for combining results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergeStrategy {
    Union,
    Intersection,
    LeftJoin,
    RightJoin,
    FullJoin,
    Custom(String),
}

/// Endpoint discovery service
#[derive(Debug, Clone)]
pub struct EndpointDiscovery {
    discovery_cache: Arc<RwLock<HashMap<String, Vec<ServiceEndpoint>>>>,
    discovery_methods: Vec<DiscoveryMethod>,
    cache_ttl: Duration,
}

/// Discovery methods for finding service endpoints
#[derive(Debug, Clone)]
pub enum DiscoveryMethod {
    Static,
    Dns,
    Consul,
    Kubernetes,
    VoID,   // Vocabulary of Interlinked Datasets
    SPARQL, // SPARQL endpoint lists
}

/// Health monitoring for service endpoints
#[derive(Debug, Clone)]
pub struct HealthMonitor {
    check_interval: Duration,
    timeout: Duration,
    failure_threshold: u32,
}

/// Service query execution request
#[derive(Debug, Clone)]
pub struct ServiceQueryRequest {
    pub service_url: String,
    pub query: String,
    pub parameters: HashMap<String, String>,
    pub timeout: Option<Duration>,
    pub headers: HashMap<String, String>,
}

/// Service query execution response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceQueryResponse {
    pub status: ResponseStatus,
    pub results: Option<serde_json::Value>,
    pub error_message: Option<String>,
    pub execution_time: Duration,
    pub endpoint_info: ServiceEndpointInfo,
}

/// Response status for service queries
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResponseStatus {
    Success,
    Timeout,
    Error,
    Retry,
}

/// Service endpoint information in response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpointInfo {
    pub url: String,
    pub response_time: Duration,
    pub attempt_count: u32,
}

/// Query cache for SERVICE requests
#[derive(Debug)]
pub struct QueryCache {
    cache: HashMap<String, CacheEntry>,
    max_size: usize,
    default_ttl: Duration,
}

/// Cache entry with TTL and access tracking
#[derive(Debug, Clone)]
struct CacheEntry {
    result: serde_json::Value,
    expires_at: Instant,
    access_count: u64,
    last_accessed: Instant,
}

/// Load balancing strategies for SERVICE endpoints
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    WeightedRoundRobin,
    ResponseTime,
    HealthScore,
    Adaptive,
}

/// Load balancer for SERVICE endpoints
#[derive(Debug, Clone)]
pub struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    endpoint_weights: HashMap<String, f64>,
    endpoint_health_scores: HashMap<String, f64>,
    round_robin_index: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Default for ServiceDelegationManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ServiceDelegationManager {
    pub fn new() -> Self {
        Self {
            endpoints: Arc::new(RwLock::new(HashMap::new())),
            executor: ParallelServiceExecutor::new(),
            merger: ServiceResultMerger::new(),
            discovery: EndpointDiscovery::new(),
            health_monitor: HealthMonitor::new(),
            query_cache: Arc::new(RwLock::new(QueryCache::new(1000, Duration::from_secs(300)))),
            load_balancer: LoadBalancer::new(LoadBalancingStrategy::Adaptive),
        }
    }

    /// Register a service endpoint
    pub async fn register_endpoint(&self, endpoint: ServiceEndpoint) -> FusekiResult<()> {
        let mut endpoints = self.endpoints.write().await;
        endpoints.insert(endpoint.url.clone(), endpoint);
        Ok(())
    }

    /// Process SERVICE clauses in a SPARQL query
    pub async fn process_service_clauses(&self, query: &str) -> FusekiResult<String> {
        let service_clauses = self.extract_service_clauses(query)?;
        let mut processed_query = query.to_string();

        for service_clause in service_clauses {
            let optimized_clause = self.optimize_service_clause(&service_clause).await?;
            processed_query = processed_query.replace(&service_clause, &optimized_clause);
        }

        Ok(processed_query)
    }

    /// Extract SERVICE clauses from query
    fn extract_service_clauses(&self, query: &str) -> FusekiResult<Vec<String>> {
        let mut clauses = Vec::new();
        let mut pos = 0;

        while let Some(service_pos) = query[pos..].find("SERVICE") {
            let abs_pos = pos + service_pos;
            if let Some(clause) = self.extract_complete_service_clause(&query[abs_pos..]) {
                let clause_len = clause.len();
                clauses.push(clause);
                pos = abs_pos + clause_len;
            } else {
                pos = abs_pos + 7; // Move past "SERVICE"
            }
        }

        Ok(clauses)
    }

    /// Extract complete SERVICE clause
    fn extract_complete_service_clause(&self, text: &str) -> Option<String> {
        // Find the service URL and the query block
        let service_start = text.find("SERVICE")?;
        let url_start = text[service_start..].find('<')?;
        let _url_end = text[service_start + url_start..].find('>')?;

        let block_start = text[service_start..].find('{')?;
        let mut brace_count = 0;
        let mut block_end = block_start;

        for (i, ch) in text[service_start + block_start..].char_indices() {
            match ch {
                '{' => brace_count += 1,
                '}' => {
                    brace_count -= 1;
                    if brace_count == 0 {
                        block_end = block_start + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }

        if block_end > block_start {
            Some(text[service_start..service_start + block_end].to_string())
        } else {
            None
        }
    }

    /// Optimize a SERVICE clause
    async fn optimize_service_clause(&self, service_clause: &str) -> FusekiResult<String> {
        // Parse the SERVICE clause
        let (service_url, inner_query) = self.parse_service_clause(service_clause)?;

        // Check if endpoint is available
        let endpoint = self.get_or_discover_endpoint(&service_url).await?;

        // Optimize the inner query for the specific endpoint
        let optimized_inner = self.optimize_for_endpoint(&inner_query, &endpoint).await?;

        // Reconstruct the SERVICE clause
        Ok(format!("SERVICE <{service_url}> {{ {optimized_inner} }}"))
    }

    /// Parse SERVICE clause to extract URL and inner query
    fn parse_service_clause(&self, service_clause: &str) -> FusekiResult<(String, String)> {
        let url_start = service_clause.find('<').ok_or_else(|| {
            crate::error::FusekiError::query_parsing("Invalid SERVICE clause - missing URL")
        })? + 1;

        let url_end = service_clause.find('>').ok_or_else(|| {
            crate::error::FusekiError::query_parsing("Invalid SERVICE clause - malformed URL")
        })?;

        let service_url = service_clause[url_start..url_end].to_string();

        let block_start = service_clause.find('{').ok_or_else(|| {
            crate::error::FusekiError::query_parsing("Invalid SERVICE clause - missing query block")
        })? + 1;

        let block_end = service_clause.rfind('}').ok_or_else(|| {
            crate::error::FusekiError::query_parsing(
                "Invalid SERVICE clause - unclosed query block",
            )
        })?;

        let inner_query = service_clause[block_start..block_end].trim().to_string();

        Ok((service_url, inner_query))
    }

    /// Get endpoint or discover it
    async fn get_or_discover_endpoint(&self, url: &str) -> FusekiResult<ServiceEndpoint> {
        // Check if we already have this endpoint
        {
            let endpoints = self.endpoints.read().await;
            if let Some(endpoint) = endpoints.get(url) {
                return Ok(endpoint.clone());
            }
        }

        // Try to discover the endpoint
        let discovered = self.discovery.discover_endpoint(url).await?;

        // Register the discovered endpoint
        self.register_endpoint(discovered.clone()).await?;

        Ok(discovered)
    }

    /// Optimize query for specific endpoint
    async fn optimize_for_endpoint(
        &self,
        query: &str,
        endpoint: &ServiceEndpoint,
    ) -> FusekiResult<String> {
        let mut optimized = query.to_string();

        // Apply endpoint-specific optimizations
        if endpoint.supported_features.contains("SPARQL_1_1") {
            optimized = self.apply_sparql_11_optimizations(&optimized)?;
        }

        if endpoint.supported_features.contains("BIND") {
            optimized = self.optimize_bind_for_endpoint(&optimized)?;
        }

        if endpoint.supported_features.contains("VALUES") {
            optimized = self.optimize_values_for_endpoint(&optimized)?;
        }

        Ok(optimized)
    }

    /// Apply SPARQL 1.1 specific optimizations
    fn apply_sparql_11_optimizations(&self, query: &str) -> FusekiResult<String> {
        // Can use advanced SPARQL 1.1 features
        Ok(query.to_string())
    }

    /// Optimize BIND clauses for endpoint
    fn optimize_bind_for_endpoint(&self, query: &str) -> FusekiResult<String> {
        // Some endpoints have better BIND support
        Ok(query.to_string())
    }

    /// Optimize VALUES clauses for endpoint
    fn optimize_values_for_endpoint(&self, query: &str) -> FusekiResult<String> {
        // Some endpoints have optimized VALUES handling
        Ok(query.to_string())
    }

    /// Execute query against multiple services
    pub async fn execute_federated_query(
        &self,
        requests: Vec<ServiceQueryRequest>,
    ) -> FusekiResult<Vec<ServiceQueryResponse>> {
        self.executor.execute_parallel(requests).await
    }

    /// Merge results from multiple services
    pub async fn merge_service_results(
        &self,
        responses: Vec<ServiceQueryResponse>,
        strategy: Option<MergeStrategy>,
    ) -> FusekiResult<serde_json::Value> {
        self.merger.merge_results(responses, strategy).await
    }

    /// Get endpoint health status
    pub async fn get_endpoint_health(&self, url: &str) -> Option<ServiceHealth> {
        let endpoints = self.endpoints.read().await;
        endpoints.get(url).map(|e| e.health_status.clone())
    }

    /// Update endpoint health status
    pub async fn update_endpoint_health(
        &self,
        url: &str,
        health: ServiceHealth,
    ) -> FusekiResult<()> {
        let mut endpoints = self.endpoints.write().await;
        if let Some(endpoint) = endpoints.get_mut(url) {
            endpoint.health_status = health;
            endpoint.last_checked = Some(std::time::SystemTime::now());
        }
        Ok(())
    }

    /// Execute SERVICE query with caching and load balancing
    pub async fn execute_service_query_optimized(
        &self,
        service_url: &str,
        query: &str,
        use_cache: bool,
    ) -> FusekiResult<ServiceQueryResponse> {
        // Generate cache key
        let cache_key = format!("{service_url}:{query}");

        // Check cache first if enabled
        if use_cache {
            let mut cache = self.query_cache.write().await;
            if let Some(cached_result) = cache.get(&cache_key) {
                debug!("Cache hit for SERVICE query: {}", service_url);
                return Ok(ServiceQueryResponse {
                    status: ResponseStatus::Success,
                    results: Some(cached_result),
                    error_message: None,
                    execution_time: Duration::from_millis(1), // Cache access time
                    endpoint_info: ServiceEndpointInfo {
                        url: service_url.to_string(),
                        response_time: Duration::from_millis(1),
                        attempt_count: 1,
                    },
                });
            }
        }

        // Get available endpoints for this URL
        let endpoints = self.get_available_endpoints(service_url).await?;

        // Use load balancer to select best endpoint
        let selected_endpoint = if endpoints.len() > 1 {
            self.load_balancer.select_endpoint(&endpoints)
        } else {
            endpoints.first()
        };

        let endpoint = selected_endpoint
            .ok_or_else(|| crate::error::FusekiError::bad_request("No available endpoints"))?;

        // Execute the query
        let request = ServiceQueryRequest {
            service_url: endpoint.url.clone(),
            query: query.to_string(),
            parameters: HashMap::new(),
            timeout: Some(endpoint.timeout),
            headers: HashMap::new(),
        };

        let response = ParallelServiceExecutor::execute_single_request(
            request,
            RetryPolicy {
                max_retries: endpoint.retry_count,
                initial_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(5),
                backoff_multiplier: 2.0,
            },
            endpoint.timeout,
        )
        .await;

        // Cache successful results
        if use_cache && response.status == ResponseStatus::Success {
            if let Some(ref result) = response.results {
                let mut cache = self.query_cache.write().await;
                cache.put(cache_key, result.clone(), None);
                debug!("Cached SERVICE query result for: {}", service_url);
            }
        }

        Ok(response)
    }

    /// Get available endpoints for a service URL
    async fn get_available_endpoints(
        &self,
        service_url: &str,
    ) -> FusekiResult<Vec<ServiceEndpoint>> {
        let endpoints = self.endpoints.read().await;

        // Get direct endpoint if available
        if let Some(endpoint) = endpoints.get(service_url) {
            if endpoint.health_status != ServiceHealth::Unhealthy {
                return Ok(vec![endpoint.clone()]);
            }
        }

        // Try to discover alternatives
        let discovered = self.discovery.discover_endpoint(service_url).await?;
        Ok(vec![discovered])
    }

    /// Update load balancer weights and health scores
    pub async fn update_load_balancer_metrics(
        &mut self,
        url: &str,
        response_time: Duration,
        success: bool,
    ) {
        // Update health score based on success rate
        let health_score = if success { 1.0 } else { 0.0 };
        self.load_balancer
            .set_endpoint_health_score(url.to_string(), health_score);

        // Update weight based on response time (faster = higher weight)
        let weight = if response_time.as_millis() > 0 {
            1000.0 / response_time.as_millis() as f64
        } else {
            10.0
        };
        self.load_balancer
            .set_endpoint_weight(url.to_string(), weight);
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> CacheStats {
        let cache = self.query_cache.read().await;
        cache.stats()
    }

    /// Get endpoint statistics
    pub fn get_endpoint_stats(&self, url: &str) -> EndpointStats {
        self.load_balancer.get_endpoint_stats(url)
    }

    /// Clear expired cache entries
    pub async fn cleanup_cache(&self) {
        let mut cache = self.query_cache.write().await;
        cache.cleanup_expired();
    }
}

impl Default for ParallelServiceExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelServiceExecutor {
    pub fn new() -> Self {
        Self {
            max_concurrent: 10,
            timeout: Duration::from_secs(30),
            retry_policy: RetryPolicy {
                max_retries: 3,
                initial_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(5),
                backoff_multiplier: 2.0,
            },
        }
    }

    /// Execute multiple service requests in parallel
    pub async fn execute_parallel(
        &self,
        requests: Vec<ServiceQueryRequest>,
    ) -> FusekiResult<Vec<ServiceQueryResponse>> {
        let mut tasks = Vec::new();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.max_concurrent));

        for request in requests {
            let semaphore = semaphore.clone();
            let retry_policy = self.retry_policy.clone();
            let timeout = self.timeout;

            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                Self::execute_single_request(request, retry_policy, timeout).await
            });

            tasks.push(task);
        }

        let mut responses = Vec::new();
        for task in tasks {
            match task.await {
                Ok(response) => responses.push(response),
                Err(e) => {
                    error!("Service execution task failed: {}", e);
                    // Add error response
                    responses.push(ServiceQueryResponse {
                        status: ResponseStatus::Error,
                        results: None,
                        error_message: Some(e.to_string()),
                        execution_time: Duration::from_secs(0),
                        endpoint_info: ServiceEndpointInfo {
                            url: "unknown".to_string(),
                            response_time: Duration::from_secs(0),
                            attempt_count: 0,
                        },
                    });
                }
            }
        }

        Ok(responses)
    }

    /// Execute a single service request with retry logic
    async fn execute_single_request(
        request: ServiceQueryRequest,
        retry_policy: RetryPolicy,
        timeout: Duration,
    ) -> ServiceQueryResponse {
        let start_time = Instant::now();
        let mut last_error = None;

        for attempt in 0..=retry_policy.max_retries {
            let attempt_start = Instant::now();

            match Self::make_http_request(&request, timeout).await {
                Ok(results) => {
                    return ServiceQueryResponse {
                        status: ResponseStatus::Success,
                        results: Some(results),
                        error_message: None,
                        execution_time: start_time.elapsed(),
                        endpoint_info: ServiceEndpointInfo {
                            url: request.service_url,
                            response_time: attempt_start.elapsed(),
                            attempt_count: attempt + 1,
                        },
                    };
                }
                Err(e) => {
                    last_error = Some(e);

                    // Don't retry on last attempt
                    if attempt < retry_policy.max_retries {
                        let delay_ms = retry_policy.initial_delay.as_millis() as f64
                            * retry_policy.backoff_multiplier.powi(attempt as i32);
                        let delay =
                            Duration::from_millis(delay_ms as u64).min(retry_policy.max_delay);
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        ServiceQueryResponse {
            status: ResponseStatus::Error,
            results: None,
            error_message: last_error.map(|e| e.to_string()),
            execution_time: start_time.elapsed(),
            endpoint_info: ServiceEndpointInfo {
                url: request.service_url,
                response_time: Duration::from_secs(0),
                attempt_count: retry_policy.max_retries + 1,
            },
        }
    }

    /// Make HTTP request to service endpoint
    async fn make_http_request(
        request: &ServiceQueryRequest,
        timeout: Duration,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        let client = reqwest::Client::builder().timeout(timeout).build()?;

        let mut http_request = client
            .post(&request.service_url)
            .header("Content-Type", "application/sparql-query")
            .header("Accept", "application/sparql-results+json,application/json")
            .body(request.query.clone());

        // Add custom headers
        for (key, value) in &request.headers {
            http_request = http_request.header(key, value);
        }

        // Add query parameters
        if !request.parameters.is_empty() {
            http_request = http_request.query(&request.parameters);
        }

        let response = http_request.send().await?;

        if !response.status().is_success() {
            return Err(format!("HTTP error: {}", response.status()).into());
        }

        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/json")
            .to_string();

        if content_type.contains("application/sparql-results+json")
            || content_type.contains("application/json")
        {
            let json_response: serde_json::Value = response.json().await?;
            Ok(json_response)
        } else {
            // Handle other content types (e.g., XML, CSV)
            let text_response = response.text().await?;
            Ok(serde_json::json!({
                "head": {
                    "vars": []
                },
                "results": {
                    "bindings": [],
                    "raw_response": text_response,
                    "content_type": content_type
                }
            }))
        }
    }
}

impl Default for ServiceResultMerger {
    fn default() -> Self {
        Self::new()
    }
}

impl ServiceResultMerger {
    pub fn new() -> Self {
        Self {
            merge_strategies: HashMap::new(),
            default_strategy: MergeStrategy::Union,
        }
    }

    /// Merge results from multiple services
    pub async fn merge_results(
        &self,
        responses: Vec<ServiceQueryResponse>,
        strategy: Option<MergeStrategy>,
    ) -> FusekiResult<serde_json::Value> {
        let strategy = strategy.unwrap_or_else(|| self.default_strategy.clone());

        let successful_responses: Vec<_> = responses
            .into_iter()
            .filter(|r| r.status == ResponseStatus::Success && r.results.is_some())
            .collect();

        if successful_responses.is_empty() {
            return Ok(serde_json::json!({
                "head": { "vars": [] },
                "results": { "bindings": [] }
            }));
        }

        match strategy {
            MergeStrategy::Union => self.merge_union(successful_responses),
            MergeStrategy::Intersection => self.merge_intersection(successful_responses),
            MergeStrategy::LeftJoin => self.merge_left_join(successful_responses),
            MergeStrategy::RightJoin => self.merge_right_join(successful_responses),
            MergeStrategy::FullJoin => self.merge_full_join(successful_responses),
            MergeStrategy::Custom(ref name) => self.merge_custom(successful_responses, name),
        }
    }

    /// Merge results using UNION strategy
    fn merge_union(&self, responses: Vec<ServiceQueryResponse>) -> FusekiResult<serde_json::Value> {
        let mut all_bindings = Vec::new();
        let mut all_vars = HashSet::new();

        for response in responses {
            if let Some(results) = response.results {
                if let Some(head) = results.get("head") {
                    if let Some(vars) = head.get("vars") {
                        if let Some(var_array) = vars.as_array() {
                            for var in var_array {
                                if let Some(var_str) = var.as_str() {
                                    all_vars.insert(var_str.to_string());
                                }
                            }
                        }
                    }
                }

                if let Some(results_obj) = results.get("results") {
                    if let Some(bindings) = results_obj.get("bindings") {
                        if let Some(bindings_array) = bindings.as_array() {
                            all_bindings.extend(bindings_array.clone());
                        }
                    }
                }
            }
        }

        let vars: Vec<_> = all_vars.into_iter().collect();

        Ok(serde_json::json!({
            "head": {
                "vars": vars
            },
            "results": {
                "bindings": all_bindings
            }
        }))
    }

    /// Merge results using INTERSECTION strategy
    fn merge_intersection(
        &self,
        responses: Vec<ServiceQueryResponse>,
    ) -> FusekiResult<serde_json::Value> {
        // Simplified intersection - would need proper implementation
        self.merge_union(responses)
    }

    /// Merge results using LEFT JOIN strategy
    fn merge_left_join(
        &self,
        responses: Vec<ServiceQueryResponse>,
    ) -> FusekiResult<serde_json::Value> {
        // Simplified left join - would need proper implementation
        self.merge_union(responses)
    }

    /// Merge results using RIGHT JOIN strategy
    fn merge_right_join(
        &self,
        responses: Vec<ServiceQueryResponse>,
    ) -> FusekiResult<serde_json::Value> {
        // Simplified right join - would need proper implementation
        self.merge_union(responses)
    }

    /// Merge results using FULL JOIN strategy
    fn merge_full_join(
        &self,
        responses: Vec<ServiceQueryResponse>,
    ) -> FusekiResult<serde_json::Value> {
        // Simplified full join - would need proper implementation
        self.merge_union(responses)
    }

    /// Merge results using custom strategy
    fn merge_custom(
        &self,
        responses: Vec<ServiceQueryResponse>,
        _strategy_name: &str,
    ) -> FusekiResult<serde_json::Value> {
        // Custom merge strategies would be implemented here
        self.merge_union(responses)
    }
}

impl Default for EndpointDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl EndpointDiscovery {
    pub fn new() -> Self {
        Self {
            discovery_cache: Arc::new(RwLock::new(HashMap::new())),
            discovery_methods: vec![
                DiscoveryMethod::Static,
                DiscoveryMethod::VoID,
                DiscoveryMethod::SPARQL,
            ],
            cache_ttl: Duration::from_secs(300), // 5 minutes
        }
    }

    /// Discover service endpoint
    pub async fn discover_endpoint(&self, url: &str) -> FusekiResult<ServiceEndpoint> {
        // Check cache first
        {
            let cache = self.discovery_cache.read().await;
            if let Some(endpoints) = cache.get(url) {
                if let Some(endpoint) = endpoints.first() {
                    return Ok(endpoint.clone());
                }
            }
        }

        // Try discovery methods
        for method in &self.discovery_methods {
            if let Ok(endpoint) = self.try_discovery_method(url, method).await {
                // Cache the result
                let mut cache = self.discovery_cache.write().await;
                cache.insert(url.to_string(), vec![endpoint.clone()]);
                return Ok(endpoint);
            }
        }

        // Fallback: create basic endpoint
        Ok(ServiceEndpoint {
            url: url.to_string(),
            name: format!("endpoint-{}", url.len()),
            supported_features: HashSet::new(),
            authentication: None,
            timeout: Duration::from_secs(30),
            retry_count: 3,
            health_status: ServiceHealth::Unknown,
            response_time_avg: None,
            last_checked: None,
        })
    }

    /// Try a specific discovery method
    async fn try_discovery_method(
        &self,
        url: &str,
        method: &DiscoveryMethod,
    ) -> FusekiResult<ServiceEndpoint> {
        match method {
            DiscoveryMethod::Static => self.discover_static(url).await,
            DiscoveryMethod::VoID => self.discover_void(url).await,
            DiscoveryMethod::SPARQL => self.discover_sparql(url).await,
            _ => Err(crate::error::FusekiError::server_error(
                "Discovery method not implemented",
            )),
        }
    }

    /// Static discovery (configuration-based)
    async fn discover_static(&self, _url: &str) -> FusekiResult<ServiceEndpoint> {
        // Would read from static configuration
        Err(crate::error::FusekiError::server_error(
            "Static discovery not configured",
        ))
    }

    /// VoID discovery (Vocabulary of Interlinked Datasets)
    async fn discover_void(&self, url: &str) -> FusekiResult<ServiceEndpoint> {
        // Would query VoID description
        Ok(ServiceEndpoint {
            url: url.to_string(),
            name: "void-discovered".to_string(),
            supported_features: ["SPARQL_1_1".to_string()].into_iter().collect(),
            authentication: None,
            timeout: Duration::from_secs(30),
            retry_count: 3,
            health_status: ServiceHealth::Unknown,
            response_time_avg: None,
            last_checked: None,
        })
    }

    /// SPARQL-based discovery
    async fn discover_sparql(&self, url: &str) -> FusekiResult<ServiceEndpoint> {
        // Would query service description
        Ok(ServiceEndpoint {
            url: url.to_string(),
            name: "sparql-discovered".to_string(),
            supported_features: ["SPARQL_1_1".to_string(), "BIND".to_string()]
                .into_iter()
                .collect(),
            authentication: None,
            timeout: Duration::from_secs(30),
            retry_count: 3,
            health_status: ServiceHealth::Unknown,
            response_time_avg: None,
            last_checked: None,
        })
    }
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthMonitor {
    pub fn new() -> Self {
        Self {
            check_interval: Duration::from_secs(60),
            timeout: Duration::from_secs(10),
            failure_threshold: 3,
        }
    }

    /// Start health monitoring for all endpoints
    pub async fn start_monitoring(&self, manager: Arc<ServiceDelegationManager>) {
        let mut interval = tokio::time::interval(self.check_interval);

        loop {
            interval.tick().await;
            self.check_all_endpoints(&manager).await;
        }
    }

    /// Check health of all registered endpoints
    async fn check_all_endpoints(&self, manager: &ServiceDelegationManager) {
        let endpoints = manager.endpoints.read().await;

        for (url, _endpoint) in endpoints.iter() {
            match self.check_endpoint_health(url).await {
                Ok(health) => {
                    if let Err(e) = manager.update_endpoint_health(url, health).await {
                        error!("Failed to update health for {}: {}", url, e);
                    }
                }
                Err(e) => {
                    warn!("Health check failed for {}: {}", url, e);
                    if let Err(e) = manager
                        .update_endpoint_health(url, ServiceHealth::Unhealthy)
                        .await
                    {
                        error!("Failed to update health for {}: {}", url, e);
                    }
                }
            }
        }
    }

    /// Check health of a specific endpoint
    async fn check_endpoint_health(&self, _url: &str) -> FusekiResult<ServiceHealth> {
        // Simple health check - in practice would make HTTP request
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Simulate random health status
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let health_value: f32 = rng.r#gen();

        if health_value > 0.9 {
            Ok(ServiceHealth::Healthy)
        } else if health_value > 0.7 {
            Ok(ServiceHealth::Degraded)
        } else {
            Ok(ServiceHealth::Unhealthy)
        }
    }
}

impl QueryCache {
    pub fn new(max_size: usize, default_ttl: Duration) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            default_ttl,
        }
    }

    /// Get cached result if available and not expired
    pub fn get(&mut self, key: &str) -> Option<serde_json::Value> {
        if let Some(entry) = self.cache.get_mut(key) {
            if entry.expires_at > Instant::now() {
                entry.access_count += 1;
                entry.last_accessed = Instant::now();
                return Some(entry.result.clone());
            } else {
                // Entry expired, remove it
                self.cache.remove(key);
            }
        }
        None
    }

    /// Store result in cache with TTL
    pub fn put(&mut self, key: String, result: serde_json::Value, ttl: Option<Duration>) {
        let ttl = ttl.unwrap_or(self.default_ttl);

        // If cache is full, evict LRU entry
        if self.cache.len() >= self.max_size {
            self.evict_lru();
        }

        let entry = CacheEntry {
            result,
            expires_at: Instant::now() + ttl,
            access_count: 1,
            last_accessed: Instant::now(),
        };

        self.cache.insert(key, entry);
    }

    /// Evict least recently used entry
    fn evict_lru(&mut self) {
        if let Some(lru_key) = self
            .cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(key, _)| key.clone())
        {
            self.cache.remove(&lru_key);
        }
    }

    /// Clear expired entries
    pub fn cleanup_expired(&mut self) {
        let now = Instant::now();
        self.cache.retain(|_, entry| entry.expires_at > now);
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let total_entries = self.cache.len();
        let total_accesses: u64 = self.cache.values().map(|e| e.access_count).sum();

        CacheStats {
            total_entries,
            max_size: self.max_size,
            total_accesses,
            hit_ratio: if total_accesses > 0 {
                total_accesses as f64 / (total_accesses + total_entries as u64) as f64
            } else {
                0.0
            },
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub max_size: usize,
    pub total_accesses: u64,
    pub hit_ratio: f64,
}

impl LoadBalancer {
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            endpoint_weights: HashMap::new(),
            endpoint_health_scores: HashMap::new(),
            round_robin_index: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// Select best endpoint based on strategy
    pub fn select_endpoint<'a>(
        &self,
        endpoints: &'a [ServiceEndpoint],
    ) -> Option<&'a ServiceEndpoint> {
        if endpoints.is_empty() {
            return None;
        }

        match self.strategy {
            LoadBalancingStrategy::RoundRobin => self.select_round_robin(endpoints),
            LoadBalancingStrategy::WeightedRoundRobin => {
                self.select_weighted_round_robin(endpoints)
            }
            LoadBalancingStrategy::ResponseTime => self.select_by_response_time(endpoints),
            LoadBalancingStrategy::HealthScore => self.select_by_health_score(endpoints),
            LoadBalancingStrategy::Adaptive => self.select_adaptive(endpoints),
        }
    }

    /// Round robin selection
    fn select_round_robin<'a>(
        &self,
        endpoints: &'a [ServiceEndpoint],
    ) -> Option<&'a ServiceEndpoint> {
        let index = self
            .round_robin_index
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        endpoints.get(index % endpoints.len())
    }

    /// Weighted round robin selection
    fn select_weighted_round_robin<'a>(
        &self,
        endpoints: &'a [ServiceEndpoint],
    ) -> Option<&'a ServiceEndpoint> {
        let mut total_weight = 0.0;
        for endpoint in endpoints {
            total_weight += self.endpoint_weights.get(&endpoint.url).unwrap_or(&1.0);
        }

        if total_weight == 0.0 {
            return self.select_round_robin(endpoints);
        }

        let mut random_weight = rand::random::<f64>() * total_weight;
        for endpoint in endpoints {
            let weight = self.endpoint_weights.get(&endpoint.url).unwrap_or(&1.0);
            random_weight -= weight;
            if random_weight <= 0.0 {
                return Some(endpoint);
            }
        }

        endpoints.first()
    }

    /// Select by response time
    fn select_by_response_time<'a>(
        &self,
        endpoints: &'a [ServiceEndpoint],
    ) -> Option<&'a ServiceEndpoint> {
        endpoints
            .iter()
            .filter(|e| e.health_status == ServiceHealth::Healthy)
            .min_by_key(|e| e.response_time_avg.unwrap_or(Duration::from_secs(u64::MAX)))
    }

    /// Select by health score
    fn select_by_health_score<'a>(
        &self,
        endpoints: &'a [ServiceEndpoint],
    ) -> Option<&'a ServiceEndpoint> {
        endpoints
            .iter()
            .filter(|e| e.health_status != ServiceHealth::Unhealthy)
            .max_by(|a, b| {
                let score_a = self.endpoint_health_scores.get(&a.url).unwrap_or(&0.5);
                let score_b = self.endpoint_health_scores.get(&b.url).unwrap_or(&0.5);
                score_a
                    .partial_cmp(score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Adaptive selection combining multiple factors
    fn select_adaptive<'a>(&self, endpoints: &'a [ServiceEndpoint]) -> Option<&'a ServiceEndpoint> {
        let mut best_endpoint = None;
        let mut best_score = f64::NEG_INFINITY;

        for endpoint in endpoints {
            if endpoint.health_status == ServiceHealth::Unhealthy {
                continue;
            }

            let mut score = 0.0;

            // Health score (40% weight)
            let health_score = self
                .endpoint_health_scores
                .get(&endpoint.url)
                .unwrap_or(&0.5);
            score += health_score * 0.4;

            // Response time score (30% weight)
            if let Some(response_time) = endpoint.response_time_avg {
                let response_score = 1.0 / (1.0 + response_time.as_millis() as f64 / 1000.0);
                score += response_score * 0.3;
            }

            // Weight score (20% weight)
            let weight = self.endpoint_weights.get(&endpoint.url).unwrap_or(&1.0);
            score += weight * 0.2;

            // Health status bonus (10% weight)
            let health_bonus = match endpoint.health_status {
                ServiceHealth::Healthy => 1.0,
                ServiceHealth::Degraded => 0.7,
                ServiceHealth::Maintenance => 0.3,
                ServiceHealth::Unknown => 0.5,
                ServiceHealth::Unhealthy => 0.0,
            };
            score += health_bonus * 0.1;

            if score > best_score {
                best_score = score;
                best_endpoint = Some(endpoint);
            }
        }

        best_endpoint
    }

    /// Update endpoint weight
    pub fn set_endpoint_weight(&mut self, url: String, weight: f64) {
        self.endpoint_weights.insert(url, weight.clamp(0.0, 10.0));
    }

    /// Update endpoint health score
    pub fn set_endpoint_health_score(&mut self, url: String, score: f64) {
        self.endpoint_health_scores
            .insert(url, score.clamp(0.0, 1.0));
    }

    /// Get endpoint statistics
    pub fn get_endpoint_stats(&self, url: &str) -> EndpointStats {
        EndpointStats {
            url: url.to_string(),
            weight: self.endpoint_weights.get(url).copied().unwrap_or(1.0),
            health_score: self.endpoint_health_scores.get(url).copied().unwrap_or(0.5),
        }
    }
}

/// Endpoint statistics
#[derive(Debug, Clone)]
pub struct EndpointStats {
    pub url: String,
    pub weight: f64,
    pub health_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_clause_extraction() {
        let manager = ServiceDelegationManager::new();

        let query = "SELECT ?s WHERE { SERVICE <http://example.org/sparql> { ?s ?p ?o } }";
        let clauses = manager.extract_service_clauses(query).unwrap();

        assert_eq!(clauses.len(), 1);
        assert!(clauses[0].contains("SERVICE <http://example.org/sparql>"));
    }

    #[tokio::test]
    async fn test_endpoint_registration() {
        let manager = ServiceDelegationManager::new();

        let endpoint = ServiceEndpoint {
            url: "http://example.org/sparql".to_string(),
            name: "test-endpoint".to_string(),
            supported_features: HashSet::new(),
            authentication: None,
            timeout: Duration::from_secs(30),
            retry_count: 3,
            health_status: ServiceHealth::Healthy,
            response_time_avg: None,
            last_checked: None,
        };

        manager.register_endpoint(endpoint).await.unwrap();

        let health = manager
            .get_endpoint_health("http://example.org/sparql")
            .await;
        assert_eq!(health, Some(ServiceHealth::Healthy));
    }

    #[tokio::test]
    async fn test_parallel_execution() {
        let executor = ParallelServiceExecutor::new();

        let requests = vec![
            ServiceQueryRequest {
                service_url: "http://example1.org/sparql".to_string(),
                query: "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
                parameters: HashMap::new(),
                timeout: None,
                headers: HashMap::new(),
            },
            ServiceQueryRequest {
                service_url: "http://example2.org/sparql".to_string(),
                query: "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
                parameters: HashMap::new(),
                timeout: None,
                headers: HashMap::new(),
            },
        ];

        let responses = executor.execute_parallel(requests).await.unwrap();
        assert_eq!(responses.len(), 2);
    }

    #[tokio::test]
    async fn test_result_merging() {
        let merger = ServiceResultMerger::new();

        let responses = vec![
            ServiceQueryResponse {
                status: ResponseStatus::Success,
                results: Some(serde_json::json!({
                    "head": { "vars": ["s"] },
                    "results": { "bindings": [{"s": {"type": "uri", "value": "http://example.org/1"}}] }
                })),
                error_message: None,
                execution_time: Duration::from_millis(100),
                endpoint_info: ServiceEndpointInfo {
                    url: "http://example1.org/sparql".to_string(),
                    response_time: Duration::from_millis(100),
                    attempt_count: 1,
                },
            },
            ServiceQueryResponse {
                status: ResponseStatus::Success,
                results: Some(serde_json::json!({
                    "head": { "vars": ["s"] },
                    "results": { "bindings": [{"s": {"type": "uri", "value": "http://example.org/2"}}] }
                })),
                error_message: None,
                execution_time: Duration::from_millis(150),
                endpoint_info: ServiceEndpointInfo {
                    url: "http://example2.org/sparql".to_string(),
                    response_time: Duration::from_millis(150),
                    attempt_count: 1,
                },
            },
        ];

        let merged = merger
            .merge_results(responses, Some(MergeStrategy::Union))
            .await
            .unwrap();

        let bindings = merged["results"]["bindings"].as_array().unwrap();
        assert_eq!(bindings.len(), 2);
    }
}

/// Advanced query cache for SERVICE results
#[derive(Debug, Clone)]
pub struct QueryCacheV2 {
    cache: HashMap<String, CacheEntryV2>,
    max_size: usize,
    default_ttl: Duration,
}

/// Cache entry with TTL and metadata
#[derive(Debug, Clone)]
pub struct CacheEntryV2 {
    data: serde_json::Value,
    created_at: Instant,
    ttl: Duration,
    access_count: u64,
    last_accessed: Instant,
    endpoint_url: String,
}

/// Load balancer for distributing queries across endpoints
#[derive(Debug, Clone)]
pub struct LoadBalancerV2 {
    strategy: LoadBalancingStrategyV2,
    endpoint_weights: HashMap<String, f64>,
    endpoint_health_scores: HashMap<String, f64>,
}

/// Load balancing strategies
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategyV2 {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    ResponseTime,
    HealthScore,
    Adaptive,
}

impl QueryCacheV2 {
    pub fn new(max_size: usize, default_ttl: Duration) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            default_ttl,
        }
    }

    /// Get cached result if available and not expired
    pub fn get(&mut self, query_hash: &str) -> Option<serde_json::Value> {
        if let Some(entry) = self.cache.get_mut(query_hash) {
            if entry.created_at.elapsed() < entry.ttl {
                entry.access_count += 1;
                entry.last_accessed = Instant::now();
                Some(entry.data.clone())
            } else {
                // Entry expired, remove it
                self.cache.remove(query_hash);
                None
            }
        } else {
            None
        }
    }

    /// Store result in cache
    pub fn set(
        &mut self,
        query_hash: String,
        data: serde_json::Value,
        endpoint_url: String,
        ttl: Option<Duration>,
    ) {
        // Remove oldest entries if cache is full
        if self.cache.len() >= self.max_size {
            self.evict_oldest();
        }

        let entry = CacheEntryV2 {
            data,
            created_at: Instant::now(),
            ttl: ttl.unwrap_or(self.default_ttl),
            access_count: 1,
            last_accessed: Instant::now(),
            endpoint_url,
        };

        self.cache.insert(query_hash, entry);
    }

    /// Evict oldest or least accessed entries
    fn evict_oldest(&mut self) {
        if let Some((oldest_key, _)) = self
            .cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(k, v)| (k.clone(), v.clone()))
        {
            self.cache.remove(&oldest_key);
        }
    }

    /// Generate cache key for query
    pub fn generate_cache_key(
        query: &str,
        endpoint: &str,
        parameters: &HashMap<String, String>,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        endpoint.hash(&mut hasher);
        for (k, v) in parameters {
            k.hash(&mut hasher);
            v.hash(&mut hasher);
        }
        format!("{:x}", hasher.finish())
    }

    /// Invalidate cache entries for specific endpoint
    pub fn invalidate_endpoint(&mut self, endpoint_url: &str) {
        self.cache
            .retain(|_, entry| entry.endpoint_url != endpoint_url);
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> ServiceCacheStats {
        let total_access_count = self.cache.values().map(|e| e.access_count).sum();
        let avg_age = if !self.cache.is_empty() {
            self.cache
                .values()
                .map(|e| e.created_at.elapsed().as_secs())
                .sum::<u64>()
                / self.cache.len() as u64
        } else {
            0
        };

        ServiceCacheStats {
            entries: self.cache.len(),
            total_access_count,
            avg_age_seconds: avg_age,
            hit_ratio: 0.0, // Would be calculated with hit/miss counters
        }
    }
}

/// Service cache statistics
#[derive(Debug, Clone)]
pub struct ServiceCacheStats {
    pub entries: usize,
    pub total_access_count: u64,
    pub avg_age_seconds: u64,
    pub hit_ratio: f64,
}

impl LoadBalancerV2 {
    pub fn new(strategy: LoadBalancingStrategyV2) -> Self {
        Self {
            strategy,
            endpoint_weights: HashMap::new(),
            endpoint_health_scores: HashMap::new(),
        }
    }

    /// Select best endpoint for query based on strategy
    pub fn select_endpoint(
        &self,
        available_endpoints: &[ServiceEndpoint],
        query_complexity: Option<f64>,
    ) -> Option<ServiceEndpoint> {
        if available_endpoints.is_empty() {
            return None;
        }

        match &self.strategy {
            LoadBalancingStrategyV2::RoundRobin => {
                // Simple round-robin (would need state tracking in real implementation)
                available_endpoints.first().cloned()
            }
            LoadBalancingStrategyV2::WeightedRoundRobin => {
                self.weighted_selection(available_endpoints)
            }
            LoadBalancingStrategyV2::ResponseTime => available_endpoints
                .iter()
                .min_by_key(|ep| ep.response_time_avg.unwrap_or(Duration::from_secs(999)))
                .cloned(),
            LoadBalancingStrategyV2::HealthScore => available_endpoints
                .iter()
                .filter(|ep| ep.health_status == ServiceHealth::Healthy)
                .max_by(|a, b| {
                    let score_a = self.endpoint_health_scores.get(&a.url).unwrap_or(&0.5);
                    let score_b = self.endpoint_health_scores.get(&b.url).unwrap_or(&0.5);
                    score_a
                        .partial_cmp(score_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .cloned(),
            LoadBalancingStrategyV2::LeastConnections => {
                // Select endpoint with least active connections (simplified)
                available_endpoints.first().cloned()
            }
            LoadBalancingStrategyV2::Adaptive => {
                self.adaptive_selection(available_endpoints, query_complexity)
            }
        }
    }

    /// Weighted selection based on endpoint weights
    fn weighted_selection(&self, endpoints: &[ServiceEndpoint]) -> Option<ServiceEndpoint> {
        let total_weight: f64 = endpoints
            .iter()
            .map(|ep| self.endpoint_weights.get(&ep.url).unwrap_or(&1.0))
            .sum();

        if total_weight <= 0.0 {
            return endpoints.first().cloned();
        }

        let mut rng = rand::thread_rng();
        let random_value: f64 = rng.gen_range(0.0..total_weight);
        let mut current_weight = 0.0;

        for endpoint in endpoints {
            current_weight += self.endpoint_weights.get(&endpoint.url).unwrap_or(&1.0);
            if random_value <= current_weight {
                return Some(endpoint.clone());
            }
        }

        endpoints.last().cloned()
    }

    /// Adaptive selection considering multiple factors
    fn adaptive_selection(
        &self,
        endpoints: &[ServiceEndpoint],
        query_complexity: Option<f64>,
    ) -> Option<ServiceEndpoint> {
        let mut scored_endpoints: Vec<(f64, ServiceEndpoint)> = endpoints
            .iter()
            .map(|ep| {
                let mut score = 0.0;

                // Health score (40% weight)
                match ep.health_status {
                    ServiceHealth::Healthy => score += 0.4,
                    ServiceHealth::Degraded => score += 0.2,
                    ServiceHealth::Unhealthy => score += 0.0,
                    ServiceHealth::Unknown => score += 0.1,
                    ServiceHealth::Maintenance => score += 0.0,
                }

                // Response time score (30% weight)
                if let Some(avg_time) = ep.response_time_avg {
                    let time_score = 1.0 / (1.0 + avg_time.as_secs_f64());
                    score += 0.3 * time_score;
                }

                // Feature support score (20% weight)
                if let Some(complexity) = query_complexity {
                    let feature_score = if (complexity > 0.7
                        && ep.supported_features.contains("complex_queries"))
                        || (complexity < 0.3 && ep.supported_features.contains("fast_queries"))
                    {
                        1.0
                    } else {
                        0.5
                    };
                    score += 0.2 * feature_score;
                }

                // Load balancing weight (10% weight)
                let weight = self.endpoint_weights.get(&ep.url).unwrap_or(&1.0);
                score += 0.1 * weight;

                (score, ep.clone())
            })
            .collect();

        scored_endpoints.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored_endpoints.into_iter().next().map(|(_, ep)| ep)
    }

    /// Update endpoint weight based on performance
    pub fn update_endpoint_weight(&mut self, endpoint_url: &str, performance_score: f64) {
        let current_weight = self.endpoint_weights.get(endpoint_url).unwrap_or(&1.0);
        let new_weight = (current_weight * 0.8) + (performance_score * 0.2);
        self.endpoint_weights
            .insert(endpoint_url.to_string(), new_weight.max(0.1).min(2.0));
    }

    /// Update endpoint health score
    pub fn update_health_score(&mut self, endpoint_url: &str, health_score: f64) {
        self.endpoint_health_scores
            .insert(endpoint_url.to_string(), health_score.max(0.0).min(1.0));
    }
}
