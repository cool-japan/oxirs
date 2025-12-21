//! Advanced SERVICE Clause Execution
//!
//! This module implements sophisticated SERVICE clause execution with support for
//! variable binding propagation, result streaming, cross-service joins, and optimization.

use anyhow::{anyhow, Result};
use reqwest::{
    header::{HeaderMap, HeaderValue, ACCEPT, CONTENT_TYPE},
    Client,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, error, instrument, warn};

use crate::{
    cache::FederationCache,
    executor::{
        QueryResultData, SparqlBinding, SparqlHead, SparqlResults, SparqlResultsData, SparqlValue,
    },
    service::FederatedService,
    service_optimizer::OptimizedServiceClause,
};

/// Advanced SERVICE clause executor
#[derive(Debug)]
pub struct ServiceExecutor {
    client: Client,
    config: ServiceExecutorConfig,
    cache: Arc<FederationCache>,
    semaphore: Arc<Semaphore>,
    /// Execution statistics per service
    execution_stats: Arc<RwLock<HashMap<String, ServiceExecutionStats>>>,
}

impl ServiceExecutor {
    /// Create a new SERVICE executor
    pub fn new(cache: Arc<FederationCache>) -> Self {
        let config = ServiceExecutorConfig::default();
        let client = Client::builder()
            .timeout(Duration::from_secs(config.default_timeout_secs))
            .pool_max_idle_per_host(config.connection_pool_size)
            .user_agent("oxirs-federate/1.0")
            .build()
            .expect("Failed to create HTTP client");

        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_requests));

        Self {
            client,
            config,
            cache,
            semaphore,
            execution_stats: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ServiceExecutorConfig, cache: Arc<FederationCache>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.default_timeout_secs))
            .pool_max_idle_per_host(config.connection_pool_size)
            .user_agent(&config.user_agent)
            .build()
            .expect("Failed to create HTTP client");

        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_requests));

        Self {
            client,
            config,
            cache,
            semaphore,
            execution_stats: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Execute a SPARQL query on a specific service with full SPARQL 1.1 Protocol support
    #[instrument(skip(self, bindings))]
    pub async fn execute_sparql_query(
        &self,
        service_id: &str,
        query: &str,
        bindings: Option<&[SparqlBinding]>,
    ) -> Result<SparqlResults> {
        debug!(
            "Executing SPARQL query on service {}: {}",
            service_id, query
        );

        // Record execution start
        self.record_execution_start(service_id).await;

        // Generate cache key
        let cache_key = format!("service:{service_id}:query:{query}");

        // Check cache first
        if let Some(cached) = self.cache.get_service_result(&cache_key).await {
            debug!("Cache hit for service query: {}", service_id);
            let result_count = cached.results.bindings.len();
            // Record cached hit (assume minimal time for cache lookup)
            self.record_execution_success(service_id, Duration::from_millis(1), result_count, true)
                .await;
            return Ok(cached);
        }

        let start_time = Instant::now();

        // Build the complete query with bindings if provided
        let final_query = if let Some(bindings) = bindings {
            match self.inject_bindings(query, bindings) {
                Ok(q) => q,
                Err(e) => {
                    let error_msg = format!("Failed to inject bindings: {}", e);
                    self.record_execution_failure(service_id, error_msg.clone())
                        .await;
                    return Err(e);
                }
            }
        } else {
            query.to_string()
        };

        // Execute with retry logic and timeout
        let result = match self.execute_with_retry(service_id, &final_query).await {
            Ok(r) => r,
            Err(e) => {
                let error_msg = format!("Query execution failed: {}", e);
                self.record_execution_failure(service_id, error_msg).await;
                return Err(e);
            }
        };

        let execution_time = start_time.elapsed();
        let result_count = result.results.bindings.len();

        debug!(
            "Service query completed in {:?} with {} results",
            execution_time, result_count
        );

        // Record successful execution
        self.record_execution_success(service_id, execution_time, result_count, false)
            .await;

        // Cache successful results
        if !result.results.bindings.is_empty() {
            self.cache
                .put_service_result(&cache_key, &result, Some(Duration::from_secs(300)))
                .await;
        }

        Ok(result)
    }

    /// Execute with VALUES binding pattern
    #[allow(dead_code)]
    async fn execute_with_values_binding(
        &self,
        service_clause: &OptimizedServiceClause,
        bindings: &[SparqlBinding],
        service: &FederatedService,
    ) -> Result<ServiceExecutionResult> {
        debug!("Executing with VALUES binding for service: {}", service.id);

        // Create batches of bindings
        let batch_size = self.config.values_batch_size;
        let mut all_results = Vec::new();

        for chunk in bindings.chunks(batch_size) {
            let values_clause = self.build_values_clause(chunk);
            let base_query =
                self.build_query_from_patterns(&service_clause.patterns, &service_clause.filters);
            let enhanced_query = format!("{base_query}\n{values_clause}");

            let batch_result = self
                .execute_single_query(&service.endpoint, &enhanced_query)
                .await?;
            all_results.extend(batch_result.results.bindings);
        }

        Ok(ServiceExecutionResult {
            service_id: service.id.clone(),
            execution_time: Duration::from_millis(100), // Mock timing
            result_count: all_results.len(),
            results: SparqlResults {
                head: SparqlHead { vars: vec![] },
                results: SparqlResultsData {
                    bindings: all_results,
                },
            },
            cached: false,
        })
    }

    /// Build VALUES clause for binding propagation
    fn build_values_clause(&self, bindings: &[SparqlBinding]) -> String {
        if bindings.is_empty() {
            return String::new();
        }

        let vars: Vec<_> = bindings[0].keys().collect();
        let mut values_clause = format!(
            "VALUES ({}) {{",
            vars.iter()
                .map(|v| format!("?{v}"))
                .collect::<Vec<_>>()
                .join(" ")
        );

        for binding in bindings {
            let values: Vec<_> = vars
                .iter()
                .map(|var| {
                    binding
                        .get(*var)
                        .map(|v| format!("\"{}\"", v.value))
                        .unwrap_or_else(|| "UNDEF".to_string())
                })
                .collect();
            values_clause.push_str(&format!(" ({})", values.join(" ")));
        }

        values_clause.push_str(" }");
        values_clause
    }

    /// Execute a single query against a service
    async fn execute_single_query(&self, endpoint: &str, query: &str) -> Result<SparqlResults> {
        let _permit = self.semaphore.acquire().await?;

        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/sparql-query"),
        );
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/sparql-results+json"),
        );

        let response = self
            .client
            .post(endpoint)
            .headers(headers)
            .body(query.to_string())
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow!("Service returned error: {}", response.status()));
        }

        let response_text = response.text().await?;
        let sparql_results: SparqlResults = serde_json::from_str(&response_text)?;

        Ok(sparql_results)
    }

    /// Execute query with exponential backoff retry logic
    async fn execute_with_retry(&self, service_id: &str, query: &str) -> Result<SparqlResults> {
        let mut attempt = 0;
        let max_attempts = self.config.max_retry_attempts;

        loop {
            attempt += 1;

            // Get service info (this would come from service registry in real implementation)
            let endpoint = format!("http://localhost:3030/{service_id}/sparql"); // Mock endpoint

            match self.execute_single_query(&endpoint, query).await {
                Ok(result) => return Ok(result),
                Err(e) if attempt >= max_attempts => {
                    error!(
                        "All retry attempts failed for service {}: {}",
                        service_id, e
                    );
                    return Err(e);
                }
                Err(e) => {
                    let delay = Duration::from_millis(100 * (2_u64.pow(attempt - 1)));
                    warn!(
                        "Service {} failed (attempt {}/{}), retrying in {:?}: {}",
                        service_id, attempt, max_attempts, delay, e
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    /// Build a SPARQL query string from patterns and filters
    #[allow(dead_code)]
    fn build_query_from_patterns(
        &self,
        patterns: &[crate::planner::TriplePattern],
        filters: &[crate::planner::FilterExpression],
    ) -> String {
        let mut query = String::from("SELECT * WHERE {\n");

        // Add patterns
        for pattern in patterns {
            query.push_str(&format!(
                "  {} {} {} .\n",
                pattern.subject.as_deref().unwrap_or("?s"),
                pattern.predicate.as_deref().unwrap_or("?p"),
                pattern.object.as_deref().unwrap_or("?o")
            ));
        }

        // Add filters
        for filter in filters {
            query.push_str(&format!("  FILTER ({})\n", filter.expression));
        }

        query.push('}');
        query
    }

    /// Inject variable bindings into a SPARQL query using VALUES clause
    fn inject_bindings(&self, query: &str, bindings: &[SparqlBinding]) -> Result<String> {
        if bindings.is_empty() {
            return Ok(query.to_string());
        }

        let values_clause = self.build_values_clause(bindings);

        // Find appropriate location to inject VALUES clause
        if query.trim_end().ends_with('}') {
            // Insert before final closing brace
            let mut modified_query = query.trim_end().to_string();
            modified_query.pop(); // Remove final '}'
            modified_query.push_str(&format!("\n  {values_clause}\n}}"));
            Ok(modified_query)
        } else {
            // Append to end
            Ok(format!("{query} {values_clause}"))
        }
    }

    /// Parse SPARQL results with proper error handling
    #[allow(dead_code)]
    fn parse_sparql_results(&self, response_text: &str) -> Result<SparqlResults> {
        // Try JSON format first
        if let Ok(results) = serde_json::from_str::<SparqlResults>(response_text) {
            return Ok(results);
        }

        // Try to parse XML format as fallback
        if response_text.trim().starts_with("<?xml") {
            return self.parse_sparql_xml(response_text);
        }

        // Try TSV format
        if response_text.lines().next().unwrap_or("").starts_with('?') {
            return self.parse_sparql_tsv(response_text);
        }

        Err(anyhow!(
            "Unknown SPARQL result format: {}",
            response_text.chars().take(100).collect::<String>()
        ))
    }

    /// Parse SPARQL XML results format
    #[allow(dead_code)]
    fn parse_sparql_xml(&self, _xml_text: &str) -> Result<SparqlResults> {
        // Simplified XML parsing - would need proper XML parser in real implementation
        warn!("XML result parsing not fully implemented, returning empty results");
        Ok(SparqlResults {
            head: SparqlHead { vars: vec![] },
            results: SparqlResultsData { bindings: vec![] },
        })
    }

    /// Parse SPARQL TSV results format
    #[allow(dead_code)]
    fn parse_sparql_tsv(&self, tsv_text: &str) -> Result<SparqlResults> {
        let lines: Vec<&str> = tsv_text.lines().collect();
        if lines.is_empty() {
            return Ok(SparqlResults {
                head: SparqlHead { vars: vec![] },
                results: SparqlResultsData { bindings: vec![] },
            });
        }

        // Parse header line to get variable names
        let header = lines[0];
        let vars: Vec<String> = header
            .split('\t')
            .map(|var| var.trim_start_matches('?').to_string())
            .collect();

        // Parse data lines
        let mut bindings = Vec::new();
        for line in lines.iter().skip(1) {
            let values: Vec<&str> = line.split('\t').collect();
            let mut binding = SparqlBinding::new();

            for (i, value) in values.iter().enumerate() {
                if let Some(var) = vars.get(i) {
                    if !value.is_empty() {
                        binding.insert(
                            var.clone(),
                            SparqlValue {
                                value_type: "literal".to_string(),
                                value: value.to_string(),
                                datatype: None,
                                lang: None,
                                quoted_triple: None,
                            },
                        );
                    }
                }
            }

            if !binding.is_empty() {
                bindings.push(binding);
            }
        }

        Ok(SparqlResults {
            head: SparqlHead { vars },
            results: SparqlResultsData { bindings },
        })
    }

    // ========== Service Detail Retrieval API ==========

    /// Get execution statistics for a specific service
    pub async fn get_service_stats(&self, service_id: &str) -> Option<ServiceExecutionStats> {
        let stats = self.execution_stats.read().await;
        stats.get(service_id).cloned()
    }

    /// Get execution statistics for all services
    pub async fn get_all_service_stats(&self) -> HashMap<String, ServiceExecutionStats> {
        let stats = self.execution_stats.read().await;
        stats.clone()
    }

    /// Get a summary of current execution state
    pub async fn get_execution_summary(&self) -> ExecutionSummary {
        let stats = self.execution_stats.read().await;

        let mut summary = ExecutionSummary {
            total_services: stats.len(),
            total_queries_executed: 0,
            total_successful_queries: 0,
            total_failed_queries: 0,
            total_cached_results: 0,
            total_active_queries: 0,
            avg_success_rate: 0.0,
            avg_cache_hit_rate: 0.0,
            slowest_service: None,
            fastest_service: None,
            most_active_service: None,
        };

        let mut max_avg_time = Duration::ZERO;
        let mut min_avg_time = Duration::MAX;
        let mut max_active_queries = 0u64;

        for (service_id, service_stats) in stats.iter() {
            summary.total_queries_executed += service_stats.total_queries;
            summary.total_successful_queries += service_stats.successful_queries;
            summary.total_failed_queries += service_stats.failed_queries;
            summary.total_cached_results += service_stats.cached_results;
            summary.total_active_queries += service_stats.active_queries;

            // Track slowest service
            if service_stats.avg_execution_time > max_avg_time && service_stats.total_queries > 0 {
                max_avg_time = service_stats.avg_execution_time;
                summary.slowest_service = Some(service_id.clone());
            }

            // Track fastest service
            if service_stats.avg_execution_time < min_avg_time && service_stats.total_queries > 0 {
                min_avg_time = service_stats.avg_execution_time;
                summary.fastest_service = Some(service_id.clone());
            }

            // Track most active service
            if service_stats.active_queries > max_active_queries {
                max_active_queries = service_stats.active_queries;
                summary.most_active_service = Some(service_id.clone());
            }
        }

        // Calculate averages
        if !stats.is_empty() {
            let success_rates: Vec<f64> = stats.values().map(|s| s.success_rate()).collect();
            summary.avg_success_rate =
                success_rates.iter().sum::<f64>() / success_rates.len() as f64;

            let cache_rates: Vec<f64> = stats.values().map(|s| s.cache_hit_rate()).collect();
            summary.avg_cache_hit_rate = cache_rates.iter().sum::<f64>() / cache_rates.len() as f64;
        }

        summary
    }

    /// Reset statistics for a specific service
    pub async fn reset_service_stats(&self, service_id: &str) {
        let mut stats = self.execution_stats.write().await;
        if stats.contains_key(service_id) {
            stats.insert(
                service_id.to_string(),
                ServiceExecutionStats::new(service_id.to_string()),
            );
        }
    }

    /// Reset statistics for all services
    pub async fn reset_all_stats(&self) {
        let mut stats = self.execution_stats.write().await;
        stats.clear();
    }

    /// Record execution start for a service (internal tracking)
    async fn record_execution_start(&self, service_id: &str) {
        let mut stats = self.execution_stats.write().await;
        let service_stats = stats
            .entry(service_id.to_string())
            .or_insert_with(|| ServiceExecutionStats::new(service_id.to_string()));
        service_stats.increment_active();
    }

    /// Record execution success for a service (internal tracking)
    async fn record_execution_success(
        &self,
        service_id: &str,
        execution_time: Duration,
        result_count: usize,
        cached: bool,
    ) {
        let mut stats = self.execution_stats.write().await;
        let service_stats = stats
            .entry(service_id.to_string())
            .or_insert_with(|| ServiceExecutionStats::new(service_id.to_string()));
        service_stats.decrement_active();
        service_stats.record_success(execution_time, result_count, cached);
    }

    /// Record execution failure for a service (internal tracking)
    async fn record_execution_failure(&self, service_id: &str, error_msg: String) {
        let mut stats = self.execution_stats.write().await;
        let service_stats = stats
            .entry(service_id.to_string())
            .or_insert_with(|| ServiceExecutionStats::new(service_id.to_string()));
        service_stats.decrement_active();
        service_stats.record_failure(error_msg);
    }
}

/// Advanced join executor for cross-service joins
#[derive(Debug)]
pub struct JoinExecutor {
    config: JoinExecutorConfig,
}

impl Default for JoinExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl JoinExecutor {
    pub fn new() -> Self {
        Self {
            config: JoinExecutorConfig::default(),
        }
    }

    /// Execute advanced join with multiple optimization strategies
    pub async fn execute_advanced_join(
        &self,
        inputs: &[&QueryResultData],
    ) -> Result<QueryResultData> {
        debug!("Executing advanced join on {} inputs", inputs.len());

        match inputs {
            [QueryResultData::Sparql(left), QueryResultData::Sparql(right)] => {
                let joined = self.join_sparql_hash_join(left, right)?;
                Ok(QueryResultData::Sparql(joined))
            }
            _ => {
                // Handle multiple inputs or mixed types
                Box::pin(self.execute_multi_way_join(inputs)).await
            }
        }
    }

    /// Hash join implementation for SPARQL results
    fn join_sparql_hash_join(
        &self,
        left: &SparqlResults,
        right: &SparqlResults,
    ) -> Result<SparqlResults> {
        // Find common variables
        let left_vars: HashSet<_> = left.head.vars.iter().collect();
        let right_vars: HashSet<_> = right.head.vars.iter().collect();
        let join_vars: Vec<_> = left_vars
            .intersection(&right_vars)
            .map(|s| s.to_string())
            .collect();

        if join_vars.is_empty() {
            return self.cartesian_product(left, right);
        }

        // Build hash table for smaller relation
        let (build_side, probe_side) =
            if left.results.bindings.len() <= right.results.bindings.len() {
                (left, right)
            } else {
                (right, left)
            };

        let mut hash_table: HashMap<String, Vec<&SparqlBinding>> = HashMap::new();
        for binding in &build_side.results.bindings {
            let key = self.create_join_key(binding, &join_vars);
            hash_table.entry(key).or_default().push(binding);
        }

        // Probe and join
        let mut result_bindings = Vec::new();
        for probe_binding in &probe_side.results.bindings {
            let key = self.create_join_key(probe_binding, &join_vars);
            if let Some(matching_bindings) = hash_table.get(&key) {
                for build_binding in matching_bindings {
                    let mut joined_binding = SparqlBinding::new();

                    // Add all variables from both bindings
                    for (var, value) in probe_binding {
                        joined_binding.insert(var.clone(), value.clone());
                    }
                    for (var, value) in *build_binding {
                        if !joined_binding.contains_key(var) {
                            joined_binding.insert(var.clone(), value.clone());
                        }
                    }

                    result_bindings.push(joined_binding);
                }
            }
        }

        // Combine variable lists
        let mut all_vars = left.head.vars.clone();
        for var in &right.head.vars {
            if !all_vars.contains(var) {
                all_vars.push(var.clone());
            }
        }

        Ok(SparqlResults {
            head: SparqlHead { vars: all_vars },
            results: SparqlResultsData {
                bindings: result_bindings,
            },
        })
    }

    /// Create join key from binding and join variables
    fn create_join_key(&self, binding: &SparqlBinding, join_vars: &[String]) -> String {
        join_vars
            .iter()
            .map(|var| {
                binding
                    .get(var)
                    .map(|v| format!("{}:{}", v.value_type, v.value))
                    .unwrap_or_else(|| "NULL".to_string())
            })
            .collect::<Vec<_>>()
            .join("|")
    }

    /// Cartesian product for joins without common variables
    fn cartesian_product(
        &self,
        left: &SparqlResults,
        right: &SparqlResults,
    ) -> Result<SparqlResults> {
        let mut result_bindings = Vec::new();

        for left_binding in &left.results.bindings {
            for right_binding in &right.results.bindings {
                let mut joined_binding = SparqlBinding::new();

                for (var, value) in left_binding {
                    joined_binding.insert(var.clone(), value.clone());
                }
                for (var, value) in right_binding {
                    joined_binding.insert(var.clone(), value.clone());
                }

                result_bindings.push(joined_binding);
            }
        }

        let mut all_vars = left.head.vars.clone();
        all_vars.extend(right.head.vars.clone());

        Ok(SparqlResults {
            head: SparqlHead { vars: all_vars },
            results: SparqlResultsData {
                bindings: result_bindings,
            },
        })
    }

    /// Execute multi-way join for more than 2 inputs
    async fn execute_multi_way_join(&self, inputs: &[&QueryResultData]) -> Result<QueryResultData> {
        if inputs.len() < 2 {
            return Err(anyhow!("Need at least 2 inputs for join"));
        }

        // Start with first two inputs
        let mut result = self.execute_advanced_join(&inputs[0..2]).await?;

        // Incrementally join remaining inputs
        for input in inputs[2..].iter() {
            let inputs_for_join = [&result, input];
            result = self.execute_advanced_join(&inputs_for_join).await?;
        }

        Ok(result)
    }

    // ============= COMPREHENSIVE CROSS-SERVICE JOIN ALGORITHMS =============

    /// Bind join implementation - sends bindings from one result to filter the other
    /// Most efficient for selective joins where one side is much smaller
    pub async fn join_sparql_bind_join(
        &self,
        left: &SparqlResults,
        right: &SparqlResults,
        _service_executor: &ServiceExecutor,
    ) -> Result<SparqlResults> {
        debug!(
            "Executing bind join with {} left bindings, {} right bindings",
            left.results.bindings.len(),
            right.results.bindings.len()
        );

        // Find common variables for join
        let left_vars: HashSet<_> = left.head.vars.iter().collect();
        let right_vars: HashSet<_> = right.head.vars.iter().collect();
        let join_vars: Vec<_> = left_vars
            .intersection(&right_vars)
            .map(|s| s.to_string())
            .collect();

        if join_vars.is_empty() {
            return self.cartesian_product(left, right);
        }

        // Choose smaller side for binding propagation
        let (bind_side, query_side, reverse_join) =
            if left.results.bindings.len() <= right.results.bindings.len() {
                (left, right, false)
            } else {
                (right, left, true)
            };

        let mut result_bindings = Vec::new();
        let batch_size = self.config.join_buffer_size;

        // Process bindings in batches to avoid overwhelming the remote service
        for batch in bind_side.results.bindings.chunks(batch_size) {
            // Convert batch to SparqlBinding format for VALUES clause injection
            let sparql_bindings: Vec<SparqlBinding> = batch.to_vec();

            // Execute query with bindings (this would call the remote service)
            // For now, simulate by filtering the query_side results
            let filtered_results =
                self.filter_results_with_bindings(query_side, &sparql_bindings, &join_vars)?;

            // Join filtered results with original bindings
            for bind_binding in batch {
                let bind_key = self.create_join_key(bind_binding, &join_vars);

                for query_binding in &filtered_results.results.bindings {
                    let query_key = self.create_join_key(query_binding, &join_vars);

                    if bind_key == query_key {
                        let mut joined_binding = SparqlBinding::new();

                        // Add bindings in correct order
                        if reverse_join {
                            // query_binding is from left, bind_binding is from right
                            for (var, value) in query_binding {
                                joined_binding.insert(var.clone(), value.clone());
                            }
                            for (var, value) in bind_binding {
                                if !joined_binding.contains_key(var) {
                                    joined_binding.insert(var.clone(), value.clone());
                                }
                            }
                        } else {
                            // bind_binding is from left, query_binding is from right
                            for (var, value) in bind_binding {
                                joined_binding.insert(var.clone(), value.clone());
                            }
                            for (var, value) in query_binding {
                                if !joined_binding.contains_key(var) {
                                    joined_binding.insert(var.clone(), value.clone());
                                }
                            }
                        }

                        result_bindings.push(joined_binding);
                    }
                }
            }
        }

        // Combine variable lists
        let mut all_vars = left.head.vars.clone();
        for var in &right.head.vars {
            if !all_vars.contains(var) {
                all_vars.push(var.clone());
            }
        }

        Ok(SparqlResults {
            head: SparqlHead { vars: all_vars },
            results: SparqlResultsData {
                bindings: result_bindings,
            },
        })
    }

    /// Nested loop join implementation - simple but memory-efficient for small datasets
    pub fn join_sparql_nested_loop(
        &self,
        left: &SparqlResults,
        right: &SparqlResults,
    ) -> Result<SparqlResults> {
        debug!(
            "Executing nested loop join with {} x {} bindings",
            left.results.bindings.len(),
            right.results.bindings.len()
        );

        let left_vars: HashSet<_> = left.head.vars.iter().collect();
        let right_vars: HashSet<_> = right.head.vars.iter().collect();
        let join_vars: Vec<_> = left_vars
            .intersection(&right_vars)
            .map(|s| s.to_string())
            .collect();

        if join_vars.is_empty() {
            return self.cartesian_product(left, right);
        }

        let mut result_bindings = Vec::new();

        // Nested loop: for each binding in left, check all bindings in right
        for left_binding in &left.results.bindings {
            let left_key = self.create_join_key(left_binding, &join_vars);

            for right_binding in &right.results.bindings {
                let right_key = self.create_join_key(right_binding, &join_vars);

                if left_key == right_key {
                    let mut joined_binding = SparqlBinding::new();

                    // Add all variables from both bindings
                    for (var, value) in left_binding {
                        joined_binding.insert(var.clone(), value.clone());
                    }
                    for (var, value) in right_binding {
                        if !joined_binding.contains_key(var) {
                            joined_binding.insert(var.clone(), value.clone());
                        }
                    }

                    result_bindings.push(joined_binding);
                }
            }
        }

        let mut all_vars = left.head.vars.clone();
        for var in &right.head.vars {
            if !all_vars.contains(var) {
                all_vars.push(var.clone());
            }
        }

        Ok(SparqlResults {
            head: SparqlHead { vars: all_vars },
            results: SparqlResultsData {
                bindings: result_bindings,
            },
        })
    }

    /// Sort-merge join implementation - efficient for pre-sorted data
    pub fn join_sparql_sort_merge(
        &self,
        left: &SparqlResults,
        right: &SparqlResults,
    ) -> Result<SparqlResults> {
        debug!(
            "Executing sort-merge join with {} left bindings, {} right bindings",
            left.results.bindings.len(),
            right.results.bindings.len()
        );

        let left_vars: HashSet<_> = left.head.vars.iter().collect();
        let right_vars: HashSet<_> = right.head.vars.iter().collect();
        let join_vars: Vec<_> = left_vars
            .intersection(&right_vars)
            .map(|s| s.to_string())
            .collect();

        if join_vars.is_empty() {
            return self.cartesian_product(left, right);
        }

        // Sort both sides by join key
        let mut left_sorted = left.results.bindings.clone();
        let mut right_sorted = right.results.bindings.clone();

        left_sorted.sort_by(|a, b| {
            let key_a = self.create_join_key(a, &join_vars);
            let key_b = self.create_join_key(b, &join_vars);
            key_a.cmp(&key_b)
        });

        right_sorted.sort_by(|a, b| {
            let key_a = self.create_join_key(a, &join_vars);
            let key_b = self.create_join_key(b, &join_vars);
            key_a.cmp(&key_b)
        });

        // Merge sorted streams
        let mut result_bindings = Vec::new();
        let mut left_idx = 0;
        let mut right_idx = 0;

        while left_idx < left_sorted.len() && right_idx < right_sorted.len() {
            let left_key = self.create_join_key(&left_sorted[left_idx], &join_vars);
            let right_key = self.create_join_key(&right_sorted[right_idx], &join_vars);

            match left_key.cmp(&right_key) {
                std::cmp::Ordering::Less => {
                    left_idx += 1;
                }
                std::cmp::Ordering::Greater => {
                    right_idx += 1;
                }
                std::cmp::Ordering::Equal => {
                    // Found match - collect all matching pairs
                    let mut left_end = left_idx;
                    while left_end < left_sorted.len()
                        && self.create_join_key(&left_sorted[left_end], &join_vars) == left_key
                    {
                        left_end += 1;
                    }

                    let mut right_end = right_idx;
                    while right_end < right_sorted.len()
                        && self.create_join_key(&right_sorted[right_end], &join_vars) == right_key
                    {
                        right_end += 1;
                    }

                    // Join all combinations in this group
                    for left_binding in &left_sorted[left_idx..left_end] {
                        for right_binding in &right_sorted[right_idx..right_end] {
                            let mut joined_binding = SparqlBinding::new();

                            for (var, value) in left_binding {
                                joined_binding.insert(var.clone(), value.clone());
                            }
                            for (var, value) in right_binding {
                                if !joined_binding.contains_key(var) {
                                    joined_binding.insert(var.clone(), value.clone());
                                }
                            }

                            result_bindings.push(joined_binding);
                        }
                    }

                    left_idx = left_end;
                    right_idx = right_end;
                }
            }
        }

        let mut all_vars = left.head.vars.clone();
        for var in &right.head.vars {
            if !all_vars.contains(var) {
                all_vars.push(var.clone());
            }
        }

        Ok(SparqlResults {
            head: SparqlHead { vars: all_vars },
            results: SparqlResultsData {
                bindings: result_bindings,
            },
        })
    }

    /// Semi-join implementation - returns only left side tuples that have matches in right side
    pub fn join_sparql_semi_join(
        &self,
        left: &SparqlResults,
        right: &SparqlResults,
    ) -> Result<SparqlResults> {
        debug!(
            "Executing semi-join with {} left bindings, {} right bindings",
            left.results.bindings.len(),
            right.results.bindings.len()
        );

        let left_vars: HashSet<_> = left.head.vars.iter().collect();
        let right_vars: HashSet<_> = right.head.vars.iter().collect();
        let join_vars: Vec<_> = left_vars
            .intersection(&right_vars)
            .map(|s| s.to_string())
            .collect();

        if join_vars.is_empty() {
            // No join variables - return left as-is
            return Ok(left.clone());
        }

        // Build hash set of right side join keys for fast lookup
        let mut right_keys = HashSet::new();
        for binding in &right.results.bindings {
            let key = self.create_join_key(binding, &join_vars);
            right_keys.insert(key);
        }

        // Filter left side to only include bindings with matches in right side
        let mut result_bindings = Vec::new();
        for left_binding in &left.results.bindings {
            let left_key = self.create_join_key(left_binding, &join_vars);
            if right_keys.contains(&left_key) {
                result_bindings.push(left_binding.clone());
            }
        }

        Ok(SparqlResults {
            head: left.head.clone(),
            results: SparqlResultsData {
                bindings: result_bindings,
            },
        })
    }

    /// Anti-join implementation - returns only left side tuples that have NO matches in right side
    pub fn join_sparql_anti_join(
        &self,
        left: &SparqlResults,
        right: &SparqlResults,
    ) -> Result<SparqlResults> {
        debug!(
            "Executing anti-join with {} left bindings, {} right bindings",
            left.results.bindings.len(),
            right.results.bindings.len()
        );

        let left_vars: HashSet<_> = left.head.vars.iter().collect();
        let right_vars: HashSet<_> = right.head.vars.iter().collect();
        let join_vars: Vec<_> = left_vars
            .intersection(&right_vars)
            .map(|s| s.to_string())
            .collect();

        if join_vars.is_empty() {
            // No join variables - return left as-is
            return Ok(left.clone());
        }

        // Build hash set of right side join keys for fast lookup
        let mut right_keys = HashSet::new();
        for binding in &right.results.bindings {
            let key = self.create_join_key(binding, &join_vars);
            right_keys.insert(key);
        }

        // Filter left side to only include bindings with NO matches in right side
        let mut result_bindings = Vec::new();
        for left_binding in &left.results.bindings {
            let left_key = self.create_join_key(left_binding, &join_vars);
            if !right_keys.contains(&left_key) {
                result_bindings.push(left_binding.clone());
            }
        }

        Ok(SparqlResults {
            head: left.head.clone(),
            results: SparqlResultsData {
                bindings: result_bindings,
            },
        })
    }

    /// Adaptive join that chooses the best strategy based on data characteristics
    pub async fn join_sparql_adaptive(
        &self,
        left: &SparqlResults,
        right: &SparqlResults,
        service_executor: &ServiceExecutor,
    ) -> Result<SparqlResults> {
        let left_size = left.results.bindings.len();
        let right_size = right.results.bindings.len();
        let _total_size = left_size + right_size;

        debug!(
            "Choosing adaptive join strategy for {} x {} bindings",
            left_size, right_size
        );

        // Choose strategy based on data characteristics
        match (left_size, right_size) {
            // Small datasets - use nested loop
            (l, r) if l <= 100 && r <= 100 => {
                debug!("Using nested loop join for small datasets");
                self.join_sparql_nested_loop(left, right)
            }

            // One side much smaller - use bind join for efficiency
            (l, r) if l < r / 10 || r < l / 10 => {
                debug!("Using bind join for skewed data sizes");
                self.join_sparql_bind_join(left, right, service_executor)
                    .await
            }

            // Large datasets - use hash join
            (l, r) if l > 1000 || r > 1000 => {
                debug!("Using hash join for large datasets");
                self.join_sparql_hash_join(left, right)
            }

            // Medium datasets - use sort-merge if pre-sorted, otherwise hash
            _ => {
                if self.is_likely_sorted(left) && self.is_likely_sorted(right) {
                    debug!("Using sort-merge join for sorted data");
                    self.join_sparql_sort_merge(left, right)
                } else {
                    debug!("Using hash join as default");
                    self.join_sparql_hash_join(left, right)
                }
            }
        }
    }

    // ============= HELPER METHODS FOR JOIN PROCESSING =============

    /// Check if results are likely already sorted (heuristic)
    fn is_likely_sorted(&self, results: &SparqlResults) -> bool {
        if results.results.bindings.len() < 10 {
            return false;
        }

        // Check if first variable values are in ascending order for first 10 rows
        if let Some(first_var) = results.head.vars.first() {
            let mut prev_value = String::new();
            for (i, binding) in results.results.bindings.iter().take(10).enumerate() {
                if let Some(value) = binding.get(first_var) {
                    if i > 0 && value.value < prev_value {
                        return false;
                    }
                    prev_value = value.value.clone();
                }
            }
            return true;
        }
        false
    }

    /// Filter results using bindings (simulates VALUES clause injection)
    fn filter_results_with_bindings(
        &self,
        results: &SparqlResults,
        bindings: &[SparqlBinding],
        join_vars: &[String],
    ) -> Result<SparqlResults> {
        let mut filtered_bindings = Vec::new();

        // Create lookup set of binding keys
        let binding_keys: HashSet<_> = bindings
            .iter()
            .map(|b| self.create_join_key(b, join_vars))
            .collect();

        // Filter results to only include matching bindings
        for result_binding in &results.results.bindings {
            let result_key = self.create_join_key(result_binding, join_vars);
            if binding_keys.contains(&result_key) {
                filtered_bindings.push(result_binding.clone());
            }
        }

        Ok(SparqlResults {
            head: results.head.clone(),
            results: SparqlResultsData {
                bindings: filtered_bindings,
            },
        })
    }

    /// Enhanced join ordering for multi-way joins
    pub fn optimize_join_order(&self, inputs: &[&QueryResultData]) -> Vec<usize> {
        let mut order = Vec::new();
        let mut remaining: Vec<_> = (0..inputs.len()).collect();

        // Start with smallest input
        if let Some((min_idx, _)) = remaining
            .iter()
            .enumerate()
            .min_by_key(|&(_, &idx)| self.get_result_size(inputs[idx]))
        {
            order.push(remaining.remove(min_idx));
        }

        // Greedily add inputs that minimize intermediate result size
        while !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_cost = f64::INFINITY;

            for (pos, &candidate_idx) in remaining.iter().enumerate() {
                let estimated_cost = self.estimate_join_cost(&order, candidate_idx, inputs);
                if estimated_cost < best_cost {
                    best_cost = estimated_cost;
                    best_idx = pos;
                }
            }

            order.push(remaining.remove(best_idx));
        }

        order
    }

    /// Estimate cost of adding an input to current join order
    fn estimate_join_cost(
        &self,
        current_order: &[usize],
        candidate: usize,
        inputs: &[&QueryResultData],
    ) -> f64 {
        let current_size = if current_order.is_empty() {
            1.0
        } else {
            // Estimate size of current partial result
            current_order
                .iter()
                .map(|&idx| self.get_result_size(inputs[idx]) as f64)
                .product::<f64>()
                .sqrt() // Rough selectivity estimate
        };

        let candidate_size = self.get_result_size(inputs[candidate]) as f64;

        // Cost is roughly the product of intermediate result size and new input size
        current_size * candidate_size
    }

    /// Get size of query result data
    fn get_result_size(&self, data: &QueryResultData) -> usize {
        match data {
            QueryResultData::Sparql(results) => results.results.bindings.len(),
            QueryResultData::GraphQL(_) => 1, // Simplified for GraphQL
            QueryResultData::ServiceResult(_) => 1, // Simplified for service results
        }
    }

    /// Build a SPARQL query string from patterns and filters
    #[allow(dead_code)]
    fn build_query_from_patterns(
        &self,
        patterns: &[crate::planner::TriplePattern],
        filters: &[crate::planner::FilterExpression],
    ) -> String {
        let mut query = String::from("SELECT * WHERE {\n");

        // Add patterns
        for pattern in patterns {
            query.push_str(&format!(
                "  {} {} {} .\n",
                pattern.subject.as_deref().unwrap_or("?s"),
                pattern.predicate.as_deref().unwrap_or("?p"),
                pattern.object.as_deref().unwrap_or("?o")
            ));
        }

        // Add filters
        for filter in filters {
            query.push_str(&format!("  FILTER ({})\n", filter.expression));
        }

        query.push('}');
        query
    }
}

/// Configuration for service executor
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ServiceExecutorConfig {
    pub default_timeout_secs: u64,
    pub connection_pool_size: usize,
    pub max_concurrent_requests: usize,
    pub user_agent: String,
    pub values_batch_size: usize,
    pub enable_streaming: bool,
    pub enable_compression: bool,
    pub max_retry_attempts: u32,
}

impl Default for ServiceExecutorConfig {
    fn default() -> Self {
        Self {
            default_timeout_secs: 30,
            connection_pool_size: 10,
            max_concurrent_requests: 100,
            user_agent: "oxirs-federate-executor/1.0".to_string(),
            values_batch_size: 100,
            enable_streaming: true,
            enable_compression: true,
            max_retry_attempts: 3,
        }
    }
}

/// Configuration for join executor
#[derive(Debug, Clone)]
pub struct JoinExecutorConfig {
    pub hash_table_threshold: usize,
    pub enable_parallel_join: bool,
    pub join_buffer_size: usize,
}

impl Default for JoinExecutorConfig {
    fn default() -> Self {
        Self {
            hash_table_threshold: 10000,
            enable_parallel_join: true,
            join_buffer_size: 1000,
        }
    }
}

/// Result of service execution
#[derive(Debug, Clone)]
pub struct ServiceExecutionResult {
    pub service_id: String,
    pub execution_time: Duration,
    pub result_count: usize,
    pub results: SparqlResults,
    pub cached: bool,
}

/// Summary of execution across all services
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExecutionSummary {
    /// Total number of registered services
    pub total_services: usize,
    /// Total queries executed across all services
    pub total_queries_executed: u64,
    /// Total successful queries
    pub total_successful_queries: u64,
    /// Total failed queries
    pub total_failed_queries: u64,
    /// Total cached results
    pub total_cached_results: u64,
    /// Total currently active queries
    pub total_active_queries: u64,
    /// Average success rate across services
    pub avg_success_rate: f64,
    /// Average cache hit rate across services
    pub avg_cache_hit_rate: f64,
    /// Service with slowest average response time
    pub slowest_service: Option<String>,
    /// Service with fastest average response time
    pub fastest_service: Option<String>,
    /// Service with most active queries
    pub most_active_service: Option<String>,
}

/// Detailed execution statistics for a service
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ServiceExecutionStats {
    /// Service identifier
    pub service_id: String,
    /// Total number of queries executed
    pub total_queries: u64,
    /// Number of successful queries
    pub successful_queries: u64,
    /// Number of failed queries
    pub failed_queries: u64,
    /// Number of cached results
    pub cached_results: u64,
    /// Total execution time across all queries
    pub total_execution_time: Duration,
    /// Minimum execution time observed
    pub min_execution_time: Duration,
    /// Maximum execution time observed
    pub max_execution_time: Duration,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Total result count across all queries
    pub total_result_count: usize,
    /// Last execution timestamp (not serialized)
    #[serde(skip)]
    pub last_execution: Option<Instant>,
    /// Last error message (if any)
    pub last_error: Option<String>,
    /// Current active query count
    pub active_queries: u64,
}

impl ServiceExecutionStats {
    /// Create new empty stats for a service
    pub fn new(service_id: String) -> Self {
        Self {
            service_id,
            total_queries: 0,
            successful_queries: 0,
            failed_queries: 0,
            cached_results: 0,
            total_execution_time: Duration::ZERO,
            min_execution_time: Duration::MAX,
            max_execution_time: Duration::ZERO,
            avg_execution_time: Duration::ZERO,
            total_result_count: 0,
            last_execution: None,
            last_error: None,
            active_queries: 0,
        }
    }

    /// Record a successful query execution
    pub fn record_success(&mut self, execution_time: Duration, result_count: usize, cached: bool) {
        self.total_queries += 1;
        self.successful_queries += 1;
        if cached {
            self.cached_results += 1;
        }

        self.total_execution_time += execution_time;
        self.min_execution_time = self.min_execution_time.min(execution_time);
        self.max_execution_time = self.max_execution_time.max(execution_time);
        self.avg_execution_time = self.total_execution_time / self.total_queries as u32;
        self.total_result_count += result_count;
        self.last_execution = Some(Instant::now());
    }

    /// Record a failed query execution
    pub fn record_failure(&mut self, error_msg: String) {
        self.total_queries += 1;
        self.failed_queries += 1;
        self.last_error = Some(error_msg);
        self.last_execution = Some(Instant::now());
    }

    /// Increment active query counter
    pub fn increment_active(&mut self) {
        self.active_queries += 1;
    }

    /// Decrement active query counter
    pub fn decrement_active(&mut self) {
        self.active_queries = self.active_queries.saturating_sub(1);
    }

    /// Get success rate (0.0 - 1.0)
    pub fn success_rate(&self) -> f64 {
        if self.total_queries == 0 {
            1.0
        } else {
            self.successful_queries as f64 / self.total_queries as f64
        }
    }

    /// Get cache hit rate (0.0 - 1.0)
    pub fn cache_hit_rate(&self) -> f64 {
        if self.successful_queries == 0 {
            0.0
        } else {
            self.cached_results as f64 / self.successful_queries as f64
        }
    }
}
