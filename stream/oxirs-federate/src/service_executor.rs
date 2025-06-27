//! Advanced SERVICE Clause Execution
//!
//! This module implements sophisticated SERVICE clause execution with support for
//! variable binding propagation, result streaming, cross-service joins, and optimization.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::{stream, Stream, StreamExt, TryStreamExt};
use reqwest::{
    header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE},
    Client, Response,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock, Semaphore};
use tokio::time::timeout;
use tracing::{debug, error, info, instrument, warn};

use crate::{
    cache::FederationCache,
    executor::{QueryResultData, SparqlBinding, SparqlResults},
    service::{FederatedService, ServiceCapability},
    service_optimizer::{OptimizedServiceClause, ServiceExecutionStrategy},
    ServiceRegistry,
};

/// Advanced SERVICE clause executor
#[derive(Debug)]
pub struct ServiceExecutor {
    client: Client,
    config: ServiceExecutorConfig,
    cache: Arc<FederationCache>,
    semaphore: Arc<Semaphore>,
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
        }
    }

    /// Execute an optimized SERVICE clause
    #[instrument(skip(self, service_clause, bindings, registry))]
    pub async fn execute_service(
        &self,
        service_clause: &OptimizedServiceClause,
        bindings: Option<&[SparqlBinding]>,
        registry: &ServiceRegistry,
        silent: bool,
    ) -> Result<ServiceExecutionResult> {
        let start_time = Instant::now();

        // Get service details
        let service = registry
            .get_service(&service_clause.service_id)
            .ok_or_else(|| anyhow!("Service {} not found", service_clause.service_id))?;

        // Execute based on strategy
        let result = match &service_clause.strategy {
            strategy if strategy.use_values_binding && bindings.is_some() => {
                self.execute_with_values_binding(service_clause, bindings.unwrap(), &service)
                    .await
            }
            strategy if strategy.stream_results => {
                self.execute_with_streaming(service_clause, bindings, &service)
                    .await
            }
            _ => {
                self.execute_standard(service_clause, bindings, &service)
                    .await
            }
        };

        // Handle SILENT errors
        match result {
            Ok(res) => Ok(res),
            Err(e) if silent => {
                warn!("SERVICE error (silent): {}", e);
                Ok(ServiceExecutionResult {
                    bindings: vec![],
                    execution_time: start_time.elapsed(),
                    rows_returned: 0,
                    cache_hit: false,
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Execute with VALUES clause for bindings
    async fn execute_with_values_binding(
        &self,
        service_clause: &OptimizedServiceClause,
        bindings: &[SparqlBinding],
        service: &FederatedService,
    ) -> Result<ServiceExecutionResult> {
        let start_time = Instant::now();

        // Batch bindings for efficient execution
        let batches = self.batch_bindings(bindings, service_clause.strategy.batch_size);
        let mut all_results = Vec::new();

        for batch in batches {
            let query = self.build_values_query(service_clause, &batch)?;
            let batch_result = self.execute_query(&service.endpoint, &query).await?;
            all_results.extend(batch_result.results.bindings);
        }

        Ok(ServiceExecutionResult {
            bindings: all_results,
            execution_time: start_time.elapsed(),
            rows_returned: all_results.len(),
            cache_hit: false,
        })
    }

    /// Execute with result streaming
    async fn execute_with_streaming(
        &self,
        service_clause: &OptimizedServiceClause,
        bindings: Option<&[SparqlBinding]>,
        service: &FederatedService,
    ) -> Result<ServiceExecutionResult> {
        let start_time = Instant::now();

        // Build query
        let query = self.build_service_query(service_clause, bindings)?;

        // Create streaming request
        let response = self
            .client
            .post(&service.endpoint)
            .header(CONTENT_TYPE, "application/sparql-query")
            .header(ACCEPT, "application/sparql-results+json")
            .header("X-Streaming", "true")
            .body(query)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Service returned error: {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
            ));
        }

        // Stream results
        let mut stream = self.create_result_stream(response).await?;
        let mut results = Vec::new();

        while let Some(binding) = stream.next().await {
            results.push(binding?);
        }

        Ok(ServiceExecutionResult {
            bindings: results,
            execution_time: start_time.elapsed(),
            rows_returned: results.len(),
            cache_hit: false,
        })
    }

    /// Standard SERVICE execution
    async fn execute_standard(
        &self,
        service_clause: &OptimizedServiceClause,
        bindings: Option<&[SparqlBinding]>,
        service: &FederatedService,
    ) -> Result<ServiceExecutionResult> {
        let start_time = Instant::now();

        // Check cache first
        let cache_key = self.generate_cache_key(service_clause, bindings);
        if let Some(cached) = self.cache.get_service_result(&cache_key).await {
            return Ok(ServiceExecutionResult {
                bindings: cached,
                execution_time: start_time.elapsed(),
                rows_returned: cached.len(),
                cache_hit: true,
            });
        }

        // Build and execute query
        let query = self.build_service_query(service_clause, bindings)?;
        let result = self.execute_query(&service.endpoint, &query).await?;

        // Cache results if appropriate
        if self.should_cache_results(&result, &service_clause.strategy) {
            self.cache
                .put_service_result(
                    &cache_key,
                    result.results.bindings.clone(),
                    Some(Duration::from_secs(self.config.cache_ttl_secs)),
                )
                .await;
        }

        Ok(ServiceExecutionResult {
            bindings: result.results.bindings,
            execution_time: start_time.elapsed(),
            rows_returned: result.results.bindings.len(),
            cache_hit: false,
        })
    }

    /// Execute SPARQL query against endpoint
    async fn execute_query(&self, endpoint: &str, query: &str) -> Result<SparqlResults> {
        // Acquire semaphore permit for rate limiting
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
            .await
            .map_err(|e| anyhow!("HTTP request failed: {}", e))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "Service returned error: {} - {}",
                response.status(),
                error_text
            ));
        }

        let sparql_results: SparqlResults = response
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse SPARQL results: {}", e))?;

        Ok(sparql_results)
    }

    /// Build SERVICE query with optimizations
    fn build_service_query(
        &self,
        service_clause: &OptimizedServiceClause,
        bindings: Option<&[SparqlBinding]>,
    ) -> Result<String> {
        let mut query = String::new();
        query.push_str("SELECT ");

        // Add projection
        let variables = self.extract_query_variables(service_clause);
        if variables.is_empty() {
            query.push_str("* ");
        } else {
            for var in &variables {
                query.push_str(var);
                query.push(' ');
            }
        }

        query.push_str("WHERE {\n");

        // Add patterns
        for pattern in &service_clause.patterns {
            query.push_str("  ");
            query.push_str(&pattern.pattern_string);
            query.push_str(" .\n");
        }

        // Add filters
        for filter in &service_clause.filters {
            query.push_str("  FILTER (");
            query.push_str(&filter.expression);
            query.push_str(")\n");
        }

        // Add pushed filters
        for filter in &service_clause.pushed_filters {
            query.push_str("  FILTER (");
            query.push_str(&filter.expression);
            query.push_str(")\n");
        }

        query.push_str("}");

        // Add LIMIT if appropriate
        if let Some(limit) = self.config.default_result_limit {
            query.push_str(&format!(" LIMIT {}", limit));
        }

        Ok(query)
    }

    /// Build VALUES query for binding propagation
    fn build_values_query(
        &self,
        service_clause: &OptimizedServiceClause,
        bindings: &[SparqlBinding],
    ) -> Result<String> {
        let mut query = String::new();
        query.push_str("SELECT ");

        // Extract variables
        let variables = self.extract_query_variables(service_clause);
        let binding_vars: HashSet<String> = bindings
            .first()
            .map(|b| b.keys().cloned().collect())
            .unwrap_or_default();

        // Add projection
        for var in &variables {
            query.push_str(var);
            query.push(' ');
        }

        query.push_str("WHERE {\n");

        // Add VALUES clause
        if !binding_vars.is_empty() {
            query.push_str("  VALUES (");
            for var in &binding_vars {
                query.push_str("?");
                query.push_str(var);
                query.push(' ');
            }
            query.push_str(") {\n");

            // Add binding values
            for binding in bindings {
                query.push_str("    (");
                for var in &binding_vars {
                    if let Some(value) = binding.get(var) {
                        query.push_str(&self.format_sparql_value(&value));
                    } else {
                        query.push_str("UNDEF");
                    }
                    query.push(' ');
                }
                query.push_str(")\n");
            }
            query.push_str("  }\n");
        }

        // Add patterns
        for pattern in &service_clause.patterns {
            query.push_str("  ");
            query.push_str(&pattern.pattern_string);
            query.push_str(" .\n");
        }

        // Add filters
        for filter in &service_clause.filters {
            query.push_str("  FILTER (");
            query.push_str(&filter.expression);
            query.push_str(")\n");
        }

        query.push_str("}");

        Ok(query)
    }

    /// Extract variables from SERVICE clause
    fn extract_query_variables(&self, service_clause: &OptimizedServiceClause) -> Vec<String> {
        let mut variables = HashSet::new();

        // Extract from patterns
        for pattern in &service_clause.patterns {
            if pattern.subject.starts_with('?') {
                variables.insert(pattern.subject.clone());
            }
            if pattern.predicate.starts_with('?') {
                variables.insert(pattern.predicate.clone());
            }
            if pattern.object.starts_with('?') {
                variables.insert(pattern.object.clone());
            }
        }

        // Extract from filters
        for filter in &service_clause.filters {
            variables.extend(filter.variables.iter().cloned());
        }

        variables.into_iter().collect()
    }

    /// Format SPARQL value for VALUES clause
    fn format_sparql_value(&self, value: &crate::executor::SparqlValue) -> String {
        match value.value_type.as_str() {
            "uri" => format!("<{}>", value.value),
            "literal" => {
                if let Some(lang) = &value.lang {
                    format!("\"{}\"@{}", value.value, lang)
                } else if let Some(datatype) = &value.datatype {
                    format!("\"{}\"^^<{}>", value.value, datatype)
                } else {
                    format!("\"{}\"", value.value)
                }
            }
            "bnode" => format!("_:{}", value.value),
            _ => "UNDEF".to_string(),
        }
    }

    /// Batch bindings for efficient execution
    fn batch_bindings(
        &self,
        bindings: &[SparqlBinding],
        batch_size: usize,
    ) -> Vec<Vec<SparqlBinding>> {
        bindings
            .chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Generate cache key for SERVICE results
    fn generate_cache_key(
        &self,
        service_clause: &OptimizedServiceClause,
        bindings: Option<&[SparqlBinding]>,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        service_clause.service_id.hash(&mut hasher);
        service_clause.endpoint.hash(&mut hasher);

        // Hash patterns
        for pattern in &service_clause.patterns {
            pattern.pattern_string.hash(&mut hasher);
        }

        // Hash filters
        for filter in &service_clause.filters {
            filter.expression.hash(&mut hasher);
        }

        // Hash bindings if present
        if let Some(bindings) = bindings {
            bindings.len().hash(&mut hasher);
        }

        format!("service:{}:{}", service_clause.service_id, hasher.finish())
    }

    /// Check if results should be cached
    fn should_cache_results(
        &self,
        result: &SparqlResults,
        strategy: &ServiceExecutionStrategy,
    ) -> bool {
        // Don't cache large result sets
        if result.results.bindings.len() > self.config.max_cache_result_size {
            return false;
        }

        // Don't cache streaming results
        if strategy.stream_results {
            return false;
        }

        true
    }

    /// Create streaming result parser
    async fn create_result_stream(
        &self,
        response: Response,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<SparqlBinding>> + Send>>> {
        // For now, parse the entire response
        // TODO: Implement true streaming JSON parser
        let results: SparqlResults = response.json().await?;
        
        let stream = stream::iter(results.results.bindings)
            .map(Ok)
            .boxed();

        Ok(stream)
    }
}

/// Service executor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceExecutorConfig {
    /// Maximum concurrent SERVICE requests
    pub max_concurrent_requests: usize,
    
    /// Default timeout in seconds
    pub default_timeout_secs: u64,
    
    /// Connection pool size per host
    pub connection_pool_size: usize,
    
    /// Default result limit
    pub default_result_limit: Option<u64>,
    
    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,
    
    /// Maximum result size to cache
    pub max_cache_result_size: usize,
    
    /// User agent string
    pub user_agent: String,
}

impl Default for ServiceExecutorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 50,
            default_timeout_secs: 30,
            connection_pool_size: 20,
            default_result_limit: None,
            cache_ttl_secs: 300,
            max_cache_result_size: 10000,
            user_agent: "oxirs-federate/1.0".to_string(),
        }
    }
}

/// Result of SERVICE execution
#[derive(Debug, Clone)]
pub struct ServiceExecutionResult {
    /// Result bindings
    pub bindings: Vec<SparqlBinding>,
    
    /// Execution time
    pub execution_time: Duration,
    
    /// Number of rows returned
    pub rows_returned: usize,
    
    /// Whether result was from cache
    pub cache_hit: bool,
}

/// Join executor for cross-service joins
#[derive(Debug)]
pub struct JoinExecutor {
    config: JoinExecutorConfig,
}

impl JoinExecutor {
    /// Create new join executor
    pub fn new() -> Self {
        Self {
            config: JoinExecutorConfig::default(),
        }
    }

    /// Execute hash join between two result sets
    pub async fn hash_join(
        &self,
        left: Vec<SparqlBinding>,
        right: Vec<SparqlBinding>,
        join_vars: &[String],
    ) -> Result<Vec<SparqlBinding>> {
        if join_vars.is_empty() {
            return self.cartesian_product(left, right);
        }

        // Build hash table from right side
        let mut hash_table: HashMap<Vec<String>, Vec<SparqlBinding>> = HashMap::new();
        
        for binding in right {
            let key: Vec<String> = join_vars
                .iter()
                .map(|var| {
                    binding
                        .get(var)
                        .map(|v| format!("{:?}", v))
                        .unwrap_or_default()
                })
                .collect();
            
            hash_table.entry(key).or_default().push(binding);
        }

        // Probe with left side
        let mut results = Vec::new();
        
        for left_binding in left {
            let key: Vec<String> = join_vars
                .iter()
                .map(|var| {
                    left_binding
                        .get(var)
                        .map(|v| format!("{:?}", v))
                        .unwrap_or_default()
                })
                .collect();

            if let Some(right_bindings) = hash_table.get(&key) {
                for right_binding in right_bindings {
                    let mut joined = left_binding.clone();
                    joined.extend(right_binding.clone());
                    results.push(joined);
                }
            }
        }

        Ok(results)
    }

    /// Execute bind join (for selective queries)
    pub async fn bind_join(
        &self,
        left: Vec<SparqlBinding>,
        service_executor: &ServiceExecutor,
        service_clause: &OptimizedServiceClause,
        registry: &ServiceRegistry,
    ) -> Result<Vec<SparqlBinding>> {
        let mut results = Vec::new();

        // Execute SERVICE for each left binding
        for binding in left {
            match service_executor
                .execute_service(service_clause, Some(&[binding.clone()]), registry, false)
                .await
            {
                Ok(service_result) => {
                    for right_binding in service_result.bindings {
                        let mut joined = binding.clone();
                        joined.extend(right_binding);
                        results.push(joined);
                    }
                }
                Err(e) => {
                    warn!("Bind join failed for binding: {}", e);
                }
            }
        }

        Ok(results)
    }

    /// Execute nested loop join
    pub async fn nested_loop_join(
        &self,
        left: Vec<SparqlBinding>,
        right: Vec<SparqlBinding>,
        join_condition: impl Fn(&SparqlBinding, &SparqlBinding) -> bool,
    ) -> Result<Vec<SparqlBinding>> {
        let mut results = Vec::new();

        for left_binding in &left {
            for right_binding in &right {
                if join_condition(left_binding, right_binding) {
                    let mut joined = left_binding.clone();
                    joined.extend(right_binding.clone());
                    results.push(joined);
                }
            }
        }

        Ok(results)
    }

    /// Execute sort-merge join
    pub async fn sort_merge_join(
        &self,
        mut left: Vec<SparqlBinding>,
        mut right: Vec<SparqlBinding>,
        join_vars: &[String],
    ) -> Result<Vec<SparqlBinding>> {
        // Sort both sides
        let sort_key = |binding: &SparqlBinding| -> Vec<String> {
            join_vars
                .iter()
                .map(|var| {
                    binding
                        .get(var)
                        .map(|v| format!("{:?}", v))
                        .unwrap_or_default()
                })
                .collect()
        };

        left.sort_by_key(sort_key);
        right.sort_by_key(sort_key);

        // Merge
        let mut results = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < left.len() && j < right.len() {
            let left_key = sort_key(&left[i]);
            let right_key = sort_key(&right[j]);

            match left_key.cmp(&right_key) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    // Found matching keys, join all combinations
                    let mut k = j;
                    while k < right.len() && sort_key(&right[k]) == left_key {
                        let mut joined = left[i].clone();
                        joined.extend(right[k].clone());
                        results.push(joined);
                        k += 1;
                    }
                    i += 1;
                }
            }
        }

        Ok(results)
    }

    /// Cartesian product for joins without variables
    fn cartesian_product(
        &self,
        left: Vec<SparqlBinding>,
        right: Vec<SparqlBinding>,
    ) -> Result<Vec<SparqlBinding>> {
        let mut results = Vec::new();

        for left_binding in &left {
            for right_binding in &right {
                let mut joined = left_binding.clone();
                joined.extend(right_binding.clone());
                results.push(joined);
            }
        }

        Ok(results)
    }
}

/// Join executor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinExecutorConfig {
    /// Memory limit for hash tables
    pub hash_table_memory_limit: usize,
    
    /// Spill to disk threshold
    pub spill_threshold: usize,
    
    /// Join algorithm selection strategy
    pub algorithm_selection: JoinAlgorithmSelection,
}

impl Default for JoinExecutorConfig {
    fn default() -> Self {
        Self {
            hash_table_memory_limit: 1024 * 1024 * 1024, // 1GB
            spill_threshold: 1000000, // 1M rows
            algorithm_selection: JoinAlgorithmSelection::CostBased,
        }
    }
}

/// Join algorithm selection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JoinAlgorithmSelection {
    /// Always use hash join
    HashJoin,
    
    /// Always use bind join
    BindJoin,
    
    /// Always use sort-merge join
    SortMerge,
    
    /// Choose based on cost estimation
    CostBased,
}

impl Default for JoinExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_executor_config() {
        let config = ServiceExecutorConfig::default();
        assert_eq!(config.max_concurrent_requests, 50);
        assert_eq!(config.default_timeout_secs, 30);
    }

    #[tokio::test]
    async fn test_join_executor_creation() {
        let executor = JoinExecutor::new();
        
        // Test empty join
        let result = executor.hash_join(vec![], vec![], &[]).await.unwrap();
        assert!(result.is_empty());
    }
}