//! Federated query executor for parallel service execution

use futures::{
    future::{join_all, select_all},
    stream::{self, StreamExt},
};
use reqwest::Client;
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    sync::{RwLock, Semaphore},
    time::timeout,
};
use url::Url;

use oxirs_arq::{Query, QueryType};
use oxirs_core::query::QueryResults;

use crate::{
    error::{FusekiError, FusekiResult},
    federation::{
        health::HealthMonitor,
        planner::{
            ExecutionStep, ExecutionStrategy, FederatedQueryPlan, QueryPlanner, ServiceSelection,
        },
        FederationConfig,
    },
};

/// Query execution result
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Query results
    pub results: QueryResults,
    /// Execution metadata
    pub metadata: QueryMetadata,
}

/// Query execution metadata
#[derive(Debug, Clone, Default)]
pub struct QueryMetadata {
    /// Execution time
    pub execution_time: Option<Duration>,
    /// Service that executed the query
    pub service_id: Option<String>,
    /// Number of results
    pub result_count: usize,
}

impl QueryResult {
    /// Create a new empty result
    pub fn new_empty() -> Self {
        Self {
            results: QueryResults::Boolean(false),
            metadata: QueryMetadata::default(),
        }
    }

    /// Get size hint for result
    pub fn size_hint(&self) -> usize {
        self.metadata.result_count
    }
}

/// Federated query executor
pub struct FederatedExecutor {
    config: FederationConfig,
    http_client: Client,
    semaphore: Arc<Semaphore>,
    planner: Arc<QueryPlanner>,
    health_monitor: Arc<HealthMonitor>,
}

/// Execution context for a federated query
#[derive(Debug)]
struct ExecutionContext {
    /// Query plan
    plan: FederatedQueryPlan,
    /// Intermediate results
    results: HashMap<String, QueryResult>,
    /// Execution metrics
    metrics: ExecutionMetrics,
}

/// Metrics for query execution
#[derive(Debug, Default)]
struct ExecutionMetrics {
    /// Total execution time
    total_time: Option<Duration>,
    /// Time per step
    step_times: HashMap<String, Duration>,
    /// Service calls made
    service_calls: u32,
    /// Failed service calls
    failed_calls: u32,
    /// Bytes transferred
    bytes_transferred: u64,
}

impl FederatedExecutor {
    /// Create a new federated executor
    pub fn new(
        config: FederationConfig,
        planner: Arc<QueryPlanner>,
        health_monitor: Arc<HealthMonitor>,
    ) -> Self {
        let max_concurrent = config.max_concurrent_requests;

        Self {
            http_client: Client::builder()
                .timeout(config.request_timeout)
                .pool_max_idle_per_host(max_concurrent)
                .build()
                .unwrap(),
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            config,
            planner,
            health_monitor,
        }
    }

    /// Execute a federated query plan
    pub async fn execute(&self, plan: FederatedQueryPlan) -> FusekiResult<QueryResult> {
        let start = Instant::now();

        let mut context = ExecutionContext {
            plan: plan.clone(),
            results: HashMap::new(),
            metrics: ExecutionMetrics::default(),
        };

        // Execute based on strategy
        let result = match plan.strategy {
            ExecutionStrategy::Sequential => self.execute_sequential(&mut context).await?,
            ExecutionStrategy::Parallel => self.execute_parallel(&mut context).await?,
            ExecutionStrategy::Adaptive => self.execute_adaptive(&mut context).await?,
        };

        // Update metrics
        context.metrics.total_time = Some(start.elapsed());
        self.report_metrics(&context.metrics);

        Ok(result)
    }

    /// Execute steps sequentially
    async fn execute_sequential(
        &self,
        context: &mut ExecutionContext,
    ) -> FusekiResult<QueryResult> {
        let mut final_result = QueryResult::new_empty();

        // Clone the steps to avoid borrow checker issues
        let steps = context.plan.steps.clone();
        for step in &steps {
            let result = self.execute_step(step, context).await?;
            context.results.insert(step.id.clone(), result.clone());
            final_result = result;
        }

        Ok(final_result)
    }

    /// Execute steps in parallel
    async fn execute_parallel(&self, context: &mut ExecutionContext) -> FusekiResult<QueryResult> {
        // Group steps by dependencies
        let step_groups = self.group_steps_by_dependencies(&context.plan.steps);

        let mut final_result = QueryResult::new_empty();

        // Execute each group in sequence, but steps within group in parallel
        for group in step_groups {
            let mut group_results = Vec::new();

            // Execute steps in this group concurrently
            let futures = group.into_iter().map(|step| {
                let step = step.clone();
                async move {
                    // Create a temporary context for this step
                    let start = std::time::Instant::now();
                    let primary_service =
                        step.services.iter().find(|s| s.is_primary).ok_or_else(|| {
                            FusekiError::Configuration {
                                message: "No primary service for step".to_string(),
                            }
                        })?;

                    self.execute_on_service_standalone(primary_service, &step.sub_query)
                        .await
                        .map(|result| (step.id.clone(), result))
                }
            });

            let results = join_all(futures).await;

            // Process results and update main context
            for result in results {
                match result {
                    Ok((step_id, res)) => {
                        context.results.insert(step_id, res.clone());
                        group_results.push(res);
                    }
                    Err(e) => return Err(e),
                }
            }

            // Use the last result as final result (or merge if needed)
            if let Some(last_result) = group_results.last() {
                final_result = last_result.clone();
            }
        }

        Ok(final_result)
    }

    /// Execute with adaptive strategy
    async fn execute_adaptive(&self, context: &mut ExecutionContext) -> FusekiResult<QueryResult> {
        // Start with parallel, fall back to sequential on errors
        match self.execute_parallel(context).await {
            Ok(result) => Ok(result),
            Err(_) => {
                tracing::warn!("Parallel execution failed, falling back to sequential");
                context.results.clear();
                context.metrics = ExecutionMetrics::default();
                self.execute_sequential(context).await
            }
        }
    }

    /// Execute a single step
    async fn execute_step(
        &self,
        step: &ExecutionStep,
        context: &mut ExecutionContext,
    ) -> FusekiResult<QueryResult> {
        let start = Instant::now();

        // Check if we should use circuit breaker
        let primary_service = step.services.iter().find(|s| s.is_primary).ok_or_else(|| {
            FusekiError::Configuration {
                message: "No primary service for step".to_string(),
            }
        })?;

        if !self
            .health_monitor
            .should_use_service(&primary_service.service_id)
            .await
        {
            // Try fallback services
            for service in &step.services {
                if !service.is_primary
                    && self
                        .health_monitor
                        .should_use_service(&service.service_id)
                        .await
                {
                    return self
                        .execute_on_service(service, &step.sub_query, context)
                        .await;
                }
            }

            return Err(FusekiError::ServiceUnavailable {
                message: format!("All services unavailable for step {}", step.id),
            });
        }

        // Execute on primary service
        let result = self
            .execute_on_service(primary_service, &step.sub_query, context)
            .await;

        // Update metrics
        context
            .metrics
            .step_times
            .insert(step.id.clone(), start.elapsed());

        // Update planner statistics
        if let Ok(ref res) = result {
            self.planner
                .update_statistics(
                    &primary_service.service_id,
                    format!("step_{}", step.id),
                    res.size_hint(),
                    start.elapsed(),
                    true,
                )
                .await;
        }

        result
    }

    /// Execute query on a specific service (standalone version for parallel execution)
    async fn execute_on_service_standalone(
        &self,
        service: &ServiceSelection,
        query: &Query,
    ) -> FusekiResult<QueryResult> {
        // Acquire semaphore permit
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| FusekiError::QueryExecution {
                message: "Failed to acquire semaphore".to_string(),
            })?;

        // Prepare request
        let query_string = self.serialize_query(query)?;

        let response = match timeout(
            self.config.request_timeout,
            self.http_client
                .post(service.service_url.as_str())
                .header("Content-Type", "application/sparql-query")
                .header("Accept", self.get_accept_header(&query.query_type))
                .body(query_string)
                .send(),
        )
        .await
        {
            Ok(Ok(resp)) => resp,
            Ok(Err(e)) => {
                return Err(FusekiError::QueryExecution {
                    message: format!("Service request failed: {}", e),
                });
            }
            Err(_) => {
                return Err(FusekiError::QueryExecution {
                    message: "Service request timed out".to_string(),
                });
            }
        };

        if !response.status().is_success() {
            return Err(FusekiError::QueryExecution {
                message: format!("Service returned error: {}", response.status()),
            });
        }

        // Parse response based on query type
        self.parse_response(response, &query.query_type).await
    }

    /// Execute query on a specific service
    async fn execute_on_service(
        &self,
        service: &ServiceSelection,
        query: &Query,
        context: &mut ExecutionContext,
    ) -> FusekiResult<QueryResult> {
        // Acquire semaphore permit
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| FusekiError::QueryExecution {
                message: "Failed to acquire semaphore".to_string(),
            })?;

        context.metrics.service_calls += 1;

        // Prepare request
        let query_string = self.serialize_query(query)?;

        let response = match timeout(
            self.config.request_timeout,
            self.http_client
                .post(service.service_url.as_str())
                .header("Content-Type", "application/sparql-query")
                .header("Accept", self.get_accept_header(&query.query_type))
                .body(query_string)
                .send(),
        )
        .await
        {
            Ok(Ok(resp)) => resp,
            Ok(Err(e)) => {
                context.metrics.failed_calls += 1;
                return Err(FusekiError::QueryExecution {
                    message: format!("Service request failed: {}", e),
                });
            }
            Err(_) => {
                context.metrics.failed_calls += 1;
                return Err(FusekiError::QueryExecution {
                    message: "Service request timed out".to_string(),
                });
            }
        };

        if !response.status().is_success() {
            context.metrics.failed_calls += 1;
            return Err(FusekiError::QueryExecution {
                message: format!("Service returned error: {}", response.status()),
            });
        }

        // Track bytes transferred
        if let Some(len) = response.content_length() {
            context.metrics.bytes_transferred += len;
        }

        // Parse response based on query type
        self.parse_response(response, &query.query_type).await
    }

    /// Parse HTTP response into QueryResult
    async fn parse_response(
        &self,
        response: reqwest::Response,
        query_type: &QueryType,
    ) -> FusekiResult<QueryResult> {
        let response_text = response
            .text()
            .await
            .map_err(|e| FusekiError::QueryExecution {
                message: format!("Failed to read response: {}", e),
            })?;

        // Parse based on query type
        let results = match query_type {
            QueryType::Select { .. } => {
                // Parse SPARQL JSON results
                let json: serde_json::Value =
                    serde_json::from_str(&response_text).map_err(|e| {
                        FusekiError::QueryExecution {
                            message: format!("Invalid JSON response: {}", e),
                        }
                    })?;

                // Extract bindings count for metadata
                let result_count = json
                    .get("results")
                    .and_then(|r| r.get("bindings"))
                    .and_then(|b| b.as_array())
                    .map(|arr| arr.len())
                    .unwrap_or(0);

                QueryResults::Solutions(vec![]) // TODO: Parse actual bindings
            }
            QueryType::Ask { .. } => {
                let json: serde_json::Value =
                    serde_json::from_str(&response_text).map_err(|e| {
                        FusekiError::QueryExecution {
                            message: format!("Invalid JSON response: {}", e),
                        }
                    })?;

                let boolean_result = json
                    .get("boolean")
                    .and_then(|b| b.as_bool())
                    .unwrap_or(false);

                QueryResults::Boolean(boolean_result)
            }
            QueryType::Construct { .. } | QueryType::Describe { .. } => {
                // Parse N-Triples or Turtle
                QueryResults::Graph(Default::default()) // TODO: Parse actual graph
            }
        };

        Ok(QueryResult {
            results,
            metadata: QueryMetadata {
                execution_time: None, // Will be set by caller
                service_id: None,
                result_count: 0, // TODO: Set actual count
            },
        })
    }

    /// Serialize query to SPARQL string
    fn serialize_query(&self, query: &Query) -> FusekiResult<String> {
        // Use oxirs-arq's built-in query serialization
        match query.to_string() {
            query_str if !query_str.is_empty() => Ok(query_str),
            _ => Err(FusekiError::QueryExecution {
                message: "Failed to serialize query".to_string(),
            }),
        }
    }

    /// Get appropriate Accept header for query type
    fn get_accept_header(&self, query_type: &QueryType) -> &'static str {
        match query_type {
            QueryType::Select { .. } | QueryType::Ask { .. } => "application/sparql-results+json",
            QueryType::Construct { .. } | QueryType::Describe { .. } => "application/n-triples",
        }
    }

    /// Group execution steps by their dependencies
    fn group_steps_by_dependencies(&self, steps: &[ExecutionStep]) -> Vec<Vec<ExecutionStep>> {
        let mut groups = Vec::new();
        let mut remaining_steps: HashMap<String, ExecutionStep> = steps
            .iter()
            .map(|step| (step.id.clone(), step.clone()))
            .collect();
        let mut processed = std::collections::HashSet::new();

        while !remaining_steps.is_empty() {
            let mut current_group = Vec::new();

            // Find steps with no unresolved dependencies
            let ready_steps: Vec<String> = remaining_steps
                .keys()
                .filter(|step_id| {
                    remaining_steps
                        .get(*step_id)
                        .map(|step| step.dependencies.iter().all(|dep| processed.contains(dep)))
                        .unwrap_or(false)
                })
                .cloned()
                .collect();

            if ready_steps.is_empty() {
                // No more resolvable dependencies - break potential cycles
                // by taking first remaining step
                if let Some((first_id, _)) = remaining_steps.iter().next() {
                    let first_id = first_id.clone();
                    if let Some(step) = remaining_steps.remove(&first_id) {
                        current_group.push(step);
                        processed.insert(first_id);
                    }
                }
            } else {
                // Add all ready steps to current group
                for step_id in ready_steps {
                    if let Some(step) = remaining_steps.remove(&step_id) {
                        current_group.push(step);
                        processed.insert(step_id);
                    }
                }
            }

            if !current_group.is_empty() {
                groups.push(current_group);
            } else {
                // Safety break to prevent infinite loop
                break;
            }
        }

        groups
    }

    /// Report execution metrics
    fn report_metrics(&self, metrics: &ExecutionMetrics) {
        tracing::info!(
            "Federated query executed in {:?} with {} service calls ({} failed)",
            metrics.total_time.unwrap_or_default(),
            metrics.service_calls,
            metrics.failed_calls
        );

        if !metrics.step_times.is_empty() {
            tracing::debug!("Step execution times: {:?}", metrics.step_times);
        }

        if metrics.bytes_transferred > 0 {
            tracing::debug!("Bytes transferred: {}", metrics.bytes_transferred);
        }
    }
}

/// Parallel result merger for combining results from multiple services
pub struct ResultMerger {
    merge_strategy: MergeStrategy,
}

#[derive(Debug, Clone)]
pub enum MergeStrategy {
    /// Union all results (default)
    Union,
    /// Intersection of results
    Intersection,
    /// Join on specific variables
    Join(Vec<String>),
    /// Custom merge function
    Custom,
}

impl ResultMerger {
    /// Create a new result merger
    pub fn new(strategy: MergeStrategy) -> Self {
        Self {
            merge_strategy: strategy,
        }
    }

    /// Merge multiple query results
    pub async fn merge(&self, results: Vec<QueryResult>) -> FusekiResult<QueryResult> {
        if results.is_empty() {
            return Ok(QueryResult::new_empty());
        }

        if results.len() == 1 {
            return Ok(results.into_iter().next().unwrap());
        }

        match &self.merge_strategy {
            MergeStrategy::Union => self.merge_union(results).await,
            MergeStrategy::Intersection => self.merge_intersection(results).await,
            MergeStrategy::Join(vars) => self.merge_join(results, vars).await,
            MergeStrategy::Custom => Err(FusekiError::QueryExecution {
                message: "Custom merge not implemented".to_string(),
            }),
        }
    }

    /// Merge results using union
    async fn merge_union(&self, results: Vec<QueryResult>) -> FusekiResult<QueryResult> {
        // TODO: Implement proper result union
        Ok(results.into_iter().next().unwrap())
    }

    /// Merge results using intersection
    async fn merge_intersection(&self, results: Vec<QueryResult>) -> FusekiResult<QueryResult> {
        // TODO: Implement proper result intersection
        Ok(results.into_iter().next().unwrap())
    }

    /// Merge results using join
    async fn merge_join(
        &self,
        results: Vec<QueryResult>,
        vars: &[String],
    ) -> FusekiResult<QueryResult> {
        // TODO: Implement proper result join
        Ok(results.into_iter().next().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_merger() {
        let merger = ResultMerger::new(MergeStrategy::Union);
        // Test would go here with proper QueryResult implementation
    }
}
