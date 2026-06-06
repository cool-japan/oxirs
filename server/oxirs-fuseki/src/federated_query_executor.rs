//! Federated SPARQL query execution: parallel dispatch, HTTP client pooling,
//! execution strategies (parallel / sequential / adaptive), and result merging.
//!
//! Implementations split out from `federated_query_optimizer` for
//! maintainability.

use crate::{
    error::{FusekiError, FusekiResult},
    federated_query_types::*,
    metrics::MetricsService,
};
use async_trait::async_trait;
use dashmap::DashMap;
use futures::{stream::FuturesUnordered, StreamExt};
use metrics::{counter, histogram};
use reqwest::{Client, ClientBuilder};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    sync::{RwLock, Semaphore},
    time::timeout,
};

// ---------------------------------------------------------------------------
// Query ID generation
// ---------------------------------------------------------------------------

/// Generate a new pseudo-UUID query identifier as a hyphen-delimited hex
/// string.
///
/// Uses `scirs2_core::random` rather than the external `uuid` crate to keep
/// the dependency footprint small. The output format mimics the
/// `xxxxxxxx-xxxx-xxxx-xxxxxxxx` shape historically used by the optimizer
/// but is not RFC 4122 compliant; callers must treat it as an opaque opaque
/// identifier.
pub fn new_query_id() -> String {
    use scirs2_core::random::{Random, Rng};

    let mut rng = Random::seed(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0),
    );

    format!(
        "{:08x}-{:04x}-{:04x}-{:08x}",
        rng.random_range(0..u32::MAX),
        rng.random_range(0..u16::MAX as u32),
        rng.random_range(0..u16::MAX as u32),
        rng.random_range(0..u32::MAX),
    )
}

// ---------------------------------------------------------------------------
// Top-level coordinator
// ---------------------------------------------------------------------------

/// Federated query optimizer for distributed SPARQL execution
pub struct FederatedQueryOptimizer {
    /// Remote endpoint registry
    pub endpoints: Arc<RwLock<EndpointRegistry>>,
    /// Query planner for federation
    pub planner: Arc<QueryPlanner>,
    /// Cost estimator for distributed queries
    pub cost_estimator: Arc<CostEstimator>,
    /// Execution engine for federated queries
    pub executor: Arc<FederatedExecutor>,
    /// Result merger for combining distributed results
    pub merger: Arc<ResultMerger>,
    /// Metrics service
    pub metrics: Arc<MetricsService>,
}

impl FederatedQueryOptimizer {
    pub fn new(metrics: Arc<MetricsService>) -> Self {
        Self {
            endpoints: Arc::new(RwLock::new(EndpointRegistry::new())),
            planner: Arc::new(QueryPlanner::new()),
            cost_estimator: Arc::new(CostEstimator::new()),
            executor: Arc::new(FederatedExecutor::new()),
            merger: Arc::new(ResultMerger::new()),
            metrics,
        }
    }

    /// Process a federated SPARQL query
    pub async fn process_federated_query(
        &self,
        query: &str,
        timeout_ms: u64,
    ) -> FusekiResult<QueryResults> {
        let start = Instant::now();

        // Extract SERVICE clauses
        let service_patterns = self.extract_service_patterns(query)?;
        if service_patterns.is_empty() {
            return Err(FusekiError::bad_request("No SERVICE patterns found"));
        }

        // Check endpoint health
        self.check_endpoint_health(&service_patterns).await?;

        // Plan query execution
        let plan = self
            .planner
            .create_execution_plan(query, &service_patterns)
            .await?;

        // Estimate costs
        let cost_estimate = self.cost_estimator.estimate_cost(&plan).await?;
        histogram!("federated_query.estimated_cost").record(cost_estimate);

        // Execute with timeout
        let results = timeout(
            Duration::from_millis(timeout_ms),
            self.executor.execute_plan(&plan),
        )
        .await
        .map_err(|_| FusekiError::TimeoutWithMessage("Federated query timeout".into()))??;

        // Record metrics
        let duration = start.elapsed();
        histogram!("federated_query.execution_time").record(duration.as_millis() as f64);
        counter!("federated_query.total").increment(1);

        Ok(results)
    }

    /// Extract SERVICE patterns from query
    pub fn extract_service_patterns(&self, query: &str) -> FusekiResult<Vec<ServicePattern>> {
        let mut patterns = Vec::new();
        let mut in_service = false;
        let mut current_service = String::new();
        let mut brace_count = 0;
        let mut service_url = String::new();

        for line in query.lines() {
            let trimmed = line.trim();

            if trimmed.starts_with("SERVICE") {
                in_service = true;
                // Extract URL from SERVICE <url> or SERVICE SILENT <url>
                if let Some(url_start) = trimmed.find('<') {
                    if let Some(url_end) = trimmed.find('>') {
                        service_url = trimmed[url_start + 1..url_end].to_string();
                    }
                }
            }

            if in_service {
                current_service.push_str(line);
                current_service.push('\n');

                for ch in trimmed.chars() {
                    match ch {
                        '{' => brace_count += 1,
                        '}' => {
                            brace_count -= 1;
                            if brace_count == 0 {
                                patterns.push(ServicePattern {
                                    service_url: service_url.clone(),
                                    pattern: current_service.clone(),
                                    is_silent: current_service.contains("SILENT"),
                                    is_optional: false,
                                });
                                in_service = false;
                                current_service.clear();
                                service_url.clear();
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(patterns)
    }

    /// Check health of required endpoints
    async fn check_endpoint_health(&self, patterns: &[ServicePattern]) -> FusekiResult<()> {
        let endpoints = self.endpoints.read().await;
        let mut futures = FuturesUnordered::new();

        for pattern in patterns {
            let endpoint_url = pattern.service_url.clone();
            if let Some(endpoint) = endpoints.endpoints.get(&endpoint_url) {
                let health_check = endpoints.check_endpoint_health(endpoint.clone());
                futures.push(health_check);
            } else if !pattern.is_silent {
                return Err(FusekiError::bad_request(format!(
                    "Unknown endpoint: {endpoint_url}"
                )));
            }
        }

        // Wait for all health checks
        while let Some(result) = futures.next().await {
            result?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FederatedExecutor
// ---------------------------------------------------------------------------

impl Default for FederatedExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl FederatedExecutor {
    pub fn new() -> Self {
        Self {
            client_pool: Arc::new(ClientPool::new()),
            strategies: Self::create_execution_strategies(),
            semaphore: Arc::new(Semaphore::new(10)),
            retry_policy: Arc::new(RetryPolicy::default()),
        }
    }

    /// Create execution strategies
    fn create_execution_strategies() -> Vec<Arc<dyn ExecutionStrategy>> {
        vec![
            Arc::new(ParallelExecutionStrategy),
            Arc::new(SequentialExecutionStrategy),
            Arc::new(AdaptiveExecutionStrategy),
        ]
    }

    /// Execute a federated query plan
    pub async fn execute_plan(&self, plan: &ExecutionPlan) -> FusekiResult<QueryResults> {
        // Select appropriate strategy
        let strategy = self.select_strategy(plan);

        // Execute with selected strategy
        strategy.execute(plan, self).await
    }

    /// Select execution strategy
    fn select_strategy(&self, plan: &ExecutionPlan) -> Arc<dyn ExecutionStrategy> {
        for strategy in &self.strategies {
            if strategy.applicable(plan) {
                return strategy.clone();
            }
        }

        // Default to sequential
        self.strategies[1].clone()
    }

    /// Execute a single fragment
    pub async fn execute_fragment(
        &self,
        fragment: &QueryFragment,
        endpoint_url: &str,
    ) -> FusekiResult<QueryResults> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| FusekiError::internal("Semaphore error"))?;

        let client = self.client_pool.get_client(endpoint_url).await?;

        let mut retries = 0;
        loop {
            match self
                .send_query(&client, endpoint_url, &fragment.sparql)
                .await
            {
                Ok(results) => return Ok(results),
                Err(_e) if retries < self.retry_policy.max_retries => {
                    retries += 1;
                    let backoff = self.calculate_backoff(retries);
                    tokio::time::sleep(backoff).await;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Send query to endpoint
    async fn send_query(
        &self,
        client: &Client,
        endpoint_url: &str,
        query: &str,
    ) -> FusekiResult<QueryResults> {
        let response = client
            .post(endpoint_url)
            .header("Content-Type", "application/sparql-query")
            .header("Accept", "application/sparql-results+json")
            .body(query.to_string())
            .send()
            .await
            .map_err(|e| FusekiError::internal(format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(FusekiError::bad_request(format!(
                "Endpoint returned status: {}",
                response.status()
            )));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| FusekiError::internal(format!("JSON parse error: {e}")))?;

        // Parse SPARQL results
        let bindings = self.parse_sparql_results(json)?;

        Ok(QueryResults {
            bindings: bindings.clone(),
            metadata: ResultMetadata {
                total_execution_time_ms: 0,
                endpoint_times: HashMap::new(),
                result_count: bindings.len(),
                partial_results: false,
            },
        })
    }

    /// Parse SPARQL JSON results
    fn parse_sparql_results(
        &self,
        json: serde_json::Value,
    ) -> FusekiResult<Vec<HashMap<String, serde_json::Value>>> {
        let results = json
            .get("results")
            .and_then(|r| r.get("bindings"))
            .and_then(|b| b.as_array())
            .ok_or_else(|| FusekiError::internal("Invalid SPARQL results format"))?;

        let mut bindings = Vec::new();
        for result in results {
            if let Some(obj) = result.as_object() {
                let mut binding = HashMap::new();
                for (var, value) in obj {
                    binding.insert(var.clone(), value.clone());
                }
                bindings.push(binding);
            }
        }

        Ok(bindings)
    }

    /// Calculate backoff duration
    pub fn calculate_backoff(&self, attempt: u32) -> Duration {
        let backoff_ms = (self.retry_policy.initial_backoff_ms as f64
            * self.retry_policy.exponential_base.powi(attempt as i32))
            as u64;

        Duration::from_millis(backoff_ms.min(self.retry_policy.max_backoff_ms))
    }
}

// ---------------------------------------------------------------------------
// ClientPool
// ---------------------------------------------------------------------------

impl Default for ClientPool {
    fn default() -> Self {
        Self::new()
    }
}

impl ClientPool {
    pub fn new() -> Self {
        Self {
            clients: DashMap::new(),
            max_connections_per_endpoint: 10,
        }
    }

    /// Get or create client for endpoint
    pub async fn get_client(&self, endpoint_url: &str) -> FusekiResult<Client> {
        if let Some(client) = self.clients.get(endpoint_url) {
            return Ok(client.clone());
        }

        let client = ClientBuilder::new()
            .pool_max_idle_per_host(self.max_connections_per_endpoint)
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| FusekiError::internal(format!("Client creation failed: {e}")))?;

        self.clients
            .insert(endpoint_url.to_string(), client.clone());
        Ok(client)
    }
}

// ---------------------------------------------------------------------------
// RetryPolicy defaults
// ---------------------------------------------------------------------------

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 100,
            max_backoff_ms: 5000,
            exponential_base: 2.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Execution strategies
// ---------------------------------------------------------------------------

/// Parallel execution strategy
struct ParallelExecutionStrategy;

#[async_trait]
impl ExecutionStrategy for ParallelExecutionStrategy {
    fn name(&self) -> &str {
        "ParallelExecution"
    }

    fn applicable(&self, plan: &ExecutionPlan) -> bool {
        plan.fragments.len() > 1 && plan.fragments.iter().all(|f| f.dependencies.is_empty())
    }

    async fn execute(
        &self,
        plan: &ExecutionPlan,
        executor: &FederatedExecutor,
    ) -> FusekiResult<QueryResults> {
        let mut futures = FuturesUnordered::new();

        // Execute all fragments in parallel
        for fragment in &plan.fragments {
            for endpoint in &fragment.target_endpoints {
                let fragment_clone = fragment.clone();
                let endpoint_clone = endpoint.clone();
                let executor_clone = executor;

                futures.push(async move {
                    executor_clone
                        .execute_fragment(&fragment_clone, &endpoint_clone)
                        .await
                });
            }
        }

        // Collect results
        let mut all_results = Vec::new();
        while let Some(result) = futures.next().await {
            all_results.push(result?);
        }

        // Merge results
        ResultMerger::new().merge_results(all_results).await
    }
}

/// Sequential execution strategy
struct SequentialExecutionStrategy;

#[async_trait]
impl ExecutionStrategy for SequentialExecutionStrategy {
    fn name(&self) -> &str {
        "SequentialExecution"
    }

    fn applicable(&self, _plan: &ExecutionPlan) -> bool {
        true // Always applicable as fallback
    }

    async fn execute(
        &self,
        plan: &ExecutionPlan,
        executor: &FederatedExecutor,
    ) -> FusekiResult<QueryResults> {
        let mut all_results = Vec::new();

        // Execute fragments sequentially
        for fragment in &plan.fragments {
            for endpoint in &fragment.target_endpoints {
                let result = executor.execute_fragment(fragment, endpoint).await?;
                all_results.push(result);
            }
        }

        // Merge results
        ResultMerger::new().merge_results(all_results).await
    }
}

/// Adaptive execution strategy
struct AdaptiveExecutionStrategy;

#[async_trait]
impl ExecutionStrategy for AdaptiveExecutionStrategy {
    fn name(&self) -> &str {
        "AdaptiveExecution"
    }

    fn applicable(&self, plan: &ExecutionPlan) -> bool {
        plan.fragments.len() > 2
    }

    async fn execute(
        &self,
        plan: &ExecutionPlan,
        executor: &FederatedExecutor,
    ) -> FusekiResult<QueryResults> {
        // Group fragments by dependencies
        let independent: Vec<_> = plan
            .fragments
            .iter()
            .filter(|f| f.dependencies.is_empty())
            .collect();

        let dependent: Vec<_> = plan
            .fragments
            .iter()
            .filter(|f| !f.dependencies.is_empty())
            .collect();

        let mut all_results = Vec::new();

        // Execute independent fragments in parallel
        if !independent.is_empty() {
            let mut futures = FuturesUnordered::new();
            for fragment in independent {
                for endpoint in &fragment.target_endpoints {
                    let fragment_clone = fragment.clone();
                    let endpoint_clone = endpoint.clone();
                    let executor_clone = executor;

                    futures.push(async move {
                        executor_clone
                            .execute_fragment(&fragment_clone, &endpoint_clone)
                            .await
                    });
                }
            }

            while let Some(result) = futures.next().await {
                all_results.push(result?);
            }
        }

        // Execute dependent fragments sequentially
        for fragment in dependent {
            for endpoint in &fragment.target_endpoints {
                let result = executor.execute_fragment(fragment, endpoint).await?;
                all_results.push(result);
            }
        }

        // Merge results
        ResultMerger::new().merge_results(all_results).await
    }
}

// ---------------------------------------------------------------------------
// ResultMerger and merge strategies
// ---------------------------------------------------------------------------

impl Default for ResultMerger {
    fn default() -> Self {
        Self::new()
    }
}

impl ResultMerger {
    pub fn new() -> Self {
        Self {
            strategies: Self::create_merge_strategies(),
            dedup_cache: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Create merge strategies
    fn create_merge_strategies() -> HashMap<String, Arc<dyn MergeStrategy>> {
        let mut strategies = HashMap::new();
        strategies.insert(
            "union".to_string(),
            Arc::new(UnionMergeStrategy) as Arc<dyn MergeStrategy>,
        );
        strategies.insert(
            "join".to_string(),
            Arc::new(JoinMergeStrategy) as Arc<dyn MergeStrategy>,
        );
        strategies.insert(
            "distinct".to_string(),
            Arc::new(DistinctMergeStrategy) as Arc<dyn MergeStrategy>,
        );
        strategies
    }

    /// Merge multiple query results
    pub async fn merge_results(&self, results: Vec<QueryResults>) -> FusekiResult<QueryResults> {
        if results.is_empty() {
            return Ok(QueryResults {
                bindings: vec![],
                metadata: ResultMetadata {
                    total_execution_time_ms: 0,
                    endpoint_times: HashMap::new(),
                    result_count: 0,
                    partial_results: false,
                },
            });
        }

        if results.len() == 1 {
            return Ok(results
                .into_iter()
                .next()
                .expect("results should not be empty after check"));
        }

        // Default to union merge
        let strategy = self
            .strategies
            .get("union")
            .expect("union strategy should be registered");
        strategy.merge(results).await
    }
}

/// Union merge strategy
struct UnionMergeStrategy;

#[async_trait]
impl MergeStrategy for UnionMergeStrategy {
    fn name(&self) -> &str {
        "UnionMerge"
    }

    async fn merge(&self, results: Vec<QueryResults>) -> FusekiResult<QueryResults> {
        let mut merged_bindings = Vec::new();
        let mut total_time = 0;
        let mut endpoint_times = HashMap::new();

        for result in results {
            merged_bindings.extend(result.bindings);
            total_time += result.metadata.total_execution_time_ms;
            endpoint_times.extend(result.metadata.endpoint_times);
        }

        let result_count = merged_bindings.len();
        Ok(QueryResults {
            bindings: merged_bindings,
            metadata: ResultMetadata {
                total_execution_time_ms: total_time,
                endpoint_times,
                result_count,
                partial_results: false,
            },
        })
    }
}

/// Join merge strategy
struct JoinMergeStrategy;

#[async_trait]
impl MergeStrategy for JoinMergeStrategy {
    fn name(&self) -> &str {
        "JoinMerge"
    }

    async fn merge(&self, results: Vec<QueryResults>) -> FusekiResult<QueryResults> {
        if results.len() != 2 {
            return Err(FusekiError::internal("Join requires exactly 2 result sets"));
        }

        let left = &results[0];
        let right = &results[1];
        let mut joined_bindings = Vec::new();

        // Simple nested loop join
        for left_binding in &left.bindings {
            for right_binding in &right.bindings {
                // Find common variables
                let common_vars: Vec<_> = left_binding
                    .keys()
                    .filter(|k| right_binding.contains_key(*k))
                    .collect();

                // Check if common variables have same values
                let mut match_found = true;
                for var in &common_vars {
                    if left_binding.get(*var) != right_binding.get(*var) {
                        match_found = false;
                        break;
                    }
                }

                if match_found {
                    // Merge bindings
                    let mut merged = left_binding.clone();
                    for (k, v) in right_binding {
                        merged.entry(k.clone()).or_insert(v.clone());
                    }
                    joined_bindings.push(merged);
                }
            }
        }

        let result_count = joined_bindings.len();
        Ok(QueryResults {
            bindings: joined_bindings,
            metadata: ResultMetadata {
                total_execution_time_ms: left.metadata.total_execution_time_ms
                    + right.metadata.total_execution_time_ms,
                endpoint_times: {
                    let mut times = left.metadata.endpoint_times.clone();
                    times.extend(right.metadata.endpoint_times.clone());
                    times
                },
                result_count,
                partial_results: false,
            },
        })
    }
}

/// Distinct merge strategy
struct DistinctMergeStrategy;

#[async_trait]
impl MergeStrategy for DistinctMergeStrategy {
    fn name(&self) -> &str {
        "DistinctMerge"
    }

    async fn merge(&self, results: Vec<QueryResults>) -> FusekiResult<QueryResults> {
        let mut seen = HashSet::new();
        let mut distinct_bindings = Vec::new();
        let mut total_time = 0;
        let mut endpoint_times = HashMap::new();

        for result in results {
            for binding in result.bindings {
                let hash = Self::hash_binding(&binding);
                if seen.insert(hash) {
                    distinct_bindings.push(binding);
                }
            }
            total_time += result.metadata.total_execution_time_ms;
            endpoint_times.extend(result.metadata.endpoint_times);
        }

        let result_count = distinct_bindings.len();
        Ok(QueryResults {
            bindings: distinct_bindings,
            metadata: ResultMetadata {
                total_execution_time_ms: total_time,
                endpoint_times,
                result_count,
                partial_results: false,
            },
        })
    }
}

impl DistinctMergeStrategy {
    fn hash_binding(binding: &HashMap<String, serde_json::Value>) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        let mut items: Vec<_> = binding.iter().collect();
        items.sort_by_key(|(k, _)| k.as_str());

        for (k, v) in items {
            k.hash(&mut hasher);
            v.to_string().hash(&mut hasher);
        }

        hasher.finish()
    }
}
