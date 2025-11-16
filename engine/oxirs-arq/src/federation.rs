//! Federated SPARQL Query Execution
//!
//! Provides advanced federation support for SPARQL queries including:
//! - SERVICE keyword execution with connection pooling
//! - Query decomposition and distribution
//! - Result merging from multiple endpoints
//! - Endpoint discovery and capability detection
//! - Load balancing and failover

use crate::algebra::{Algebra, Binding, Expression, Solution, Term, Variable};
use anyhow::{anyhow, Result};
use dashmap::DashMap;
use reqwest;
use scirs2_core::metrics::{Counter, Timer};
use scirs2_core::random::{rng, Rng}; // Use scirs2-core for random number generation
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

/// Configuration for federated query execution
#[derive(Debug, Clone)]
pub struct FederationConfig {
    /// Maximum number of concurrent SERVICE requests
    pub max_concurrent_requests: usize,
    /// Request timeout
    pub request_timeout: Duration,
    /// Maximum number of retries for failed requests
    pub max_retries: usize,
    /// Base delay for exponential backoff
    pub retry_base_delay: Duration,
    /// Maximum retry delay
    pub retry_max_delay: Duration,
    /// Enable result caching
    pub enable_caching: bool,
    /// Cache TTL
    pub cache_ttl: Duration,
    /// Connection pool size per endpoint
    pub connection_pool_size: usize,
    /// Enable endpoint health monitoring
    pub enable_health_monitoring: bool,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Enable query decomposition
    pub enable_query_decomposition: bool,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 32,
            request_timeout: Duration::from_secs(60),
            max_retries: 3,
            retry_base_delay: Duration::from_millis(100),
            retry_max_delay: Duration::from_secs(10),
            enable_caching: true,
            cache_ttl: Duration::from_secs(300),
            connection_pool_size: 8,
            enable_health_monitoring: true,
            health_check_interval: Duration::from_secs(30),
            load_balancing: LoadBalancingStrategy::LeastLoaded,
            enable_query_decomposition: true,
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Random selection
    Random,
    /// Select least loaded endpoint
    LeastLoaded,
    /// Select endpoint with best response time
    FastestResponse,
    /// Adaptive strategy based on historical performance
    Adaptive,
}

/// Federation executor
pub struct FederationExecutor {
    config: FederationConfig,
    /// Connection pools per endpoint
    connection_pools: Arc<DashMap<String, Arc<ConnectionPool>>>,
    /// Result cache
    result_cache: Arc<DashMap<QueryCacheKey, CachedResult>>,
    /// Endpoint health status
    endpoint_health: Arc<DashMap<String, EndpointHealth>>,
    /// Metrics
    metrics: Arc<FederationMetrics>,
    /// Semaphore for limiting concurrent requests
    request_semaphore: Arc<Semaphore>,
}

impl FederationExecutor {
    /// Create a new federation executor
    pub fn new(config: FederationConfig) -> Self {
        let max_concurrent = config.max_concurrent_requests;
        Self {
            config,
            connection_pools: Arc::new(DashMap::new()),
            result_cache: Arc::new(DashMap::new()),
            endpoint_health: Arc::new(DashMap::new()),
            metrics: Arc::new(FederationMetrics::new()),
            request_semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }

    /// Execute a SERVICE query
    pub async fn execute_service(
        &self,
        endpoint: &Term,
        pattern: &Algebra,
        silent: bool,
    ) -> Result<Solution> {
        let endpoint_url = self.extract_endpoint_url(endpoint)?;

        // Check endpoint health
        if self.config.enable_health_monitoring {
            let health = self.get_endpoint_health(&endpoint_url);
            if !health.is_healthy() && !silent {
                return Err(anyhow!(
                    "Endpoint {} is unhealthy: {}",
                    endpoint_url,
                    health.status_message
                ));
            }
        }

        // Check cache
        if self.config.enable_caching {
            let cache_key = QueryCacheKey {
                endpoint: endpoint_url.clone(),
                query: format!("{:?}", pattern),
            };
            if let Some(cached) = self.result_cache.get(&cache_key) {
                if !cached.is_expired() {
                    self.metrics.cache_hits.inc();
                    return Ok(cached.results.clone());
                } else {
                    self.result_cache.remove(&cache_key);
                }
            }
            self.metrics.cache_misses.inc();
        }

        // Execute with retry logic
        let start = Instant::now();
        let result = self
            .execute_with_retry(&endpoint_url, pattern, silent)
            .await;
        let elapsed = start.elapsed();

        // Update metrics
        let _timer_guard = self.metrics.request_duration.start();
        match &result {
            Ok(results) => {
                self.metrics.successful_requests.inc();
                self.metrics.results_received.add(results.len() as u64);

                // Cache result
                if self.config.enable_caching {
                    let cache_key = QueryCacheKey {
                        endpoint: endpoint_url.clone(),
                        query: format!("{:?}", pattern),
                    };
                    self.result_cache.insert(
                        cache_key,
                        CachedResult {
                            results: results.clone(),
                            timestamp: Instant::now(),
                            ttl: self.config.cache_ttl,
                        },
                    );
                }

                // Update endpoint health
                if let Some(mut health) = self.endpoint_health.get_mut(&endpoint_url) {
                    health.record_success(elapsed);
                }
            }
            Err(_) => {
                self.metrics.failed_requests.inc();

                // Update endpoint health
                if let Some(mut health) = self.endpoint_health.get_mut(&endpoint_url) {
                    health.record_failure();
                }
            }
        }

        result
    }

    /// Execute with retry logic and exponential backoff
    async fn execute_with_retry(
        &self,
        endpoint: &str,
        pattern: &Algebra,
        silent: bool,
    ) -> Result<Solution> {
        let mut last_error = None;
        let mut delay = self.config.retry_base_delay;

        for attempt in 0..=self.config.max_retries {
            // Acquire semaphore permit
            let _permit = self
                .request_semaphore
                .acquire()
                .await
                .map_err(|e| anyhow!("Failed to acquire request semaphore: {}", e))?;

            // Get connection from pool
            let pool = self.get_or_create_pool(endpoint);

            match self.execute_query(endpoint, pattern, &pool).await {
                Ok(results) => {
                    if attempt > 0 {
                        self.metrics.retried_requests.inc();
                    }
                    return Ok(results);
                }
                Err(e) => {
                    last_error = Some(e);

                    if attempt < self.config.max_retries {
                        // Exponential backoff with jitter (using scirs2-core random for thread safety)
                        let jitter_ms = rng().random_range(0..100);
                        let jitter = Duration::from_millis(jitter_ms);
                        tokio::time::sleep(delay + jitter).await;
                        delay = (delay * 2).min(self.config.retry_max_delay);
                    }
                }
            }
        }

        if silent {
            Ok(Vec::new())
        } else {
            Err(last_error.unwrap_or_else(|| anyhow!("All retry attempts failed")))
        }
    }

    /// Convert Algebra to SPARQL query string
    fn algebra_to_sparql(&self, algebra: &Algebra) -> Result<String> {
        let mut query = String::from("SELECT * WHERE {\n");
        self.algebra_to_sparql_recursive(algebra, &mut query, 1)?;
        query.push_str("}\n");
        Ok(query)
    }

    /// Recursively convert Algebra to SPARQL
    fn algebra_to_sparql_recursive(
        &self,
        algebra: &Algebra,
        query: &mut String,
        indent: usize,
    ) -> Result<()> {
        let indent_str = "  ".repeat(indent);

        match algebra {
            Algebra::Bgp(patterns) => {
                for pattern in patterns {
                    query.push_str(&format!(
                        "{}{}  {}  {} .\n",
                        indent_str,
                        self.term_to_sparql(&pattern.subject),
                        self.term_to_sparql(&pattern.predicate),
                        self.term_to_sparql(&pattern.object)
                    ));
                }
            }
            Algebra::Join { left, right } => {
                self.algebra_to_sparql_recursive(left, query, indent)?;
                self.algebra_to_sparql_recursive(right, query, indent)?;
            }
            Algebra::LeftJoin {
                left,
                right,
                filter,
            } => {
                self.algebra_to_sparql_recursive(left, query, indent)?;
                query.push_str(&format!("{}OPTIONAL {{\n", indent_str));
                self.algebra_to_sparql_recursive(right, query, indent + 1)?;
                if let Some(expr) = filter {
                    query.push_str(&format!(
                        "{}  FILTER ({})\n",
                        indent_str,
                        self.expr_to_sparql(expr)?
                    ));
                }
                query.push_str(&format!("{}}}\n", indent_str));
            }
            Algebra::Union { left, right } => {
                query.push_str(&format!("{}{{\n", indent_str));
                self.algebra_to_sparql_recursive(left, query, indent + 1)?;
                query.push_str(&format!("{}}} UNION {{\n", indent_str));
                self.algebra_to_sparql_recursive(right, query, indent + 1)?;
                query.push_str(&format!("{}}}\n", indent_str));
            }
            Algebra::Filter { pattern, condition } => {
                self.algebra_to_sparql_recursive(pattern, query, indent)?;
                query.push_str(&format!(
                    "{}FILTER ({})\n",
                    indent_str,
                    self.expr_to_sparql(condition)?
                ));
            }
            _ => {
                // For other algebra types, use a simplified representation
                query.push_str(&format!("{}# Complex pattern: {:?}\n", indent_str, algebra));
            }
        }
        Ok(())
    }

    /// Convert Term to SPARQL representation
    fn term_to_sparql(&self, term: &Term) -> String {
        match term {
            Term::Variable(var) => format!("?{}", var.name()),
            Term::Iri(iri) => format!("<{}>", iri.as_str()),
            Term::Literal(lit) => format!("\"{}\"", lit.value),
            Term::BlankNode(bn_id) => format!("_:{}", bn_id),
            _ => "?var".to_string(),
        }
    }

    /// Convert Expression to SPARQL representation
    fn expr_to_sparql(&self, expr: &Expression) -> Result<String> {
        // Simplified expression conversion
        Ok(format!("{:?}", expr))
    }

    /// Execute query against endpoint
    async fn execute_query(
        &self,
        endpoint: &str,
        pattern: &Algebra,
        pool: &Arc<ConnectionPool>,
    ) -> Result<Solution> {
        let start_time = Instant::now();

        // Convert algebra to SPARQL query
        let sparql_query = self.algebra_to_sparql(pattern)?;

        tracing::debug!(
            "Executing federated query to {}: {}",
            endpoint,
            sparql_query
        );

        // Acquire connection from pool
        {
            let mut active = pool.active_connections.lock();
            if *active >= pool.size {
                return Err(anyhow!(
                    "Connection pool exhausted for endpoint: {}",
                    endpoint
                ));
            }
            *active += 1;
        }

        // Make HTTP request
        let result = self.execute_http_request(endpoint, &sparql_query).await;

        // Release connection
        {
            let mut active = pool.active_connections.lock();
            *active = active.saturating_sub(1);
        }

        // Update metrics
        let _duration = start_time.elapsed();
        let _guard = self.metrics.request_duration.start();

        match &result {
            Ok(solutions) => {
                self.metrics.successful_requests.inc();
                self.metrics.results_received.add(solutions.len() as u64);
                tracing::debug!("Received {} solutions from {}", solutions.len(), endpoint);
            }
            Err(e) => {
                self.metrics.failed_requests.inc();
                tracing::warn!("Failed to execute query on {}: {}", endpoint, e);
            }
        }

        result
    }

    /// Execute HTTP request to SPARQL endpoint
    async fn execute_http_request(&self, endpoint: &str, query: &str) -> Result<Solution> {
        let client = reqwest::Client::builder()
            .timeout(self.config.request_timeout)
            .build()?;

        // Make POST request with SPARQL query
        let response = client
            .post(endpoint)
            .header("Accept", "application/sparql-results+json")
            .header("Content-Type", "application/sparql-query")
            .body(query.to_string())
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "SPARQL endpoint returned error: {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
            ));
        }

        // Parse JSON response
        let json_response: serde_json::Value = response.json().await?;

        // Convert JSON results to Solution format
        self.parse_sparql_json_results(&json_response)
    }

    /// Parse SPARQL JSON results into Solution format
    fn parse_sparql_json_results(&self, json: &serde_json::Value) -> Result<Solution> {
        let bindings = json["results"]["bindings"]
            .as_array()
            .ok_or_else(|| anyhow!("Invalid SPARQL JSON results format"))?;

        let mut solutions = Vec::new();

        for binding in bindings {
            let mut solution_binding = Binding::new();

            if let Some(obj) = binding.as_object() {
                for (var_name, value) in obj {
                    // Use new_unchecked for performance since variable names from SPARQL endpoints are trusted
                    let variable = Variable::new_unchecked(var_name.clone());
                    let term = self.parse_sparql_json_term(value)?;
                    solution_binding.insert(variable, term);
                }
            }

            solutions.push(solution_binding);
        }

        Ok(solutions)
    }

    /// Parse SPARQL JSON term
    fn parse_sparql_json_term(&self, value: &serde_json::Value) -> Result<Term> {
        let term_type = value["type"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing term type"))?;

        let term_value = value["value"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing term value"))?;

        match term_type {
            "uri" => {
                use oxirs_core::model::NamedNode;
                Ok(Term::Iri(NamedNode::new_unchecked(term_value)))
            }
            "literal" => {
                use crate::algebra::Literal as AlgebraLiteral;
                use oxirs_core::model::NamedNode;
                let datatype = value.get("datatype").and_then(|v| v.as_str());
                let language = value.get("xml:lang").and_then(|v| v.as_str());

                if let Some(lang) = language {
                    Ok(Term::Literal(AlgebraLiteral {
                        value: term_value.to_string(),
                        language: Some(lang.to_string()),
                        datatype: None,
                    }))
                } else if let Some(dt) = datatype {
                    Ok(Term::Literal(AlgebraLiteral {
                        value: term_value.to_string(),
                        language: None,
                        datatype: Some(NamedNode::new_unchecked(dt)),
                    }))
                } else {
                    Ok(Term::Literal(AlgebraLiteral {
                        value: term_value.to_string(),
                        language: None,
                        datatype: None,
                    }))
                }
            }
            "bnode" => {
                // BlankNode in algebra is just a String
                Ok(Term::BlankNode(term_value.to_string()))
            }
            _ => Err(anyhow!("Unknown term type: {}", term_type)),
        }
    }

    /// Get or create connection pool for endpoint
    fn get_or_create_pool(&self, endpoint: &str) -> Arc<ConnectionPool> {
        self.connection_pools
            .entry(endpoint.to_string())
            .or_insert_with(|| {
                Arc::new(ConnectionPool::new(
                    endpoint.to_string(),
                    self.config.connection_pool_size,
                ))
            })
            .clone()
    }

    /// Get endpoint health status
    fn get_endpoint_health(&self, endpoint: &str) -> EndpointHealth {
        self.endpoint_health
            .entry(endpoint.to_string())
            .or_insert_with(EndpointHealth::new)
            .clone()
    }

    /// Extract endpoint URL from Term
    fn extract_endpoint_url(&self, term: &Term) -> Result<String> {
        match term {
            Term::Iri(iri) => Ok(iri.as_str().to_string()),
            Term::Variable(_) => Err(anyhow!("Cannot use variable as SERVICE endpoint")),
            _ => Err(anyhow!("Invalid SERVICE endpoint: {:?}", term)),
        }
    }

    /// Decompose query for federated execution
    pub fn decompose_query(&self, query: &Algebra) -> Vec<FederatedSubquery> {
        let mut subqueries = Vec::new();
        Self::extract_service_patterns(query, &mut subqueries);
        subqueries
    }

    /// Extract SERVICE patterns from query
    fn extract_service_patterns(algebra: &Algebra, subqueries: &mut Vec<FederatedSubquery>) {
        match algebra {
            Algebra::Service {
                endpoint,
                pattern,
                silent,
            } => {
                subqueries.push(FederatedSubquery {
                    endpoint: endpoint.clone(),
                    pattern: (**pattern).clone(),
                    silent: *silent,
                    dependencies: Vec::new(),
                });
            }
            Algebra::Join { left, right }
            | Algebra::Union { left, right }
            | Algebra::Minus { left, right } => {
                Self::extract_service_patterns(left, subqueries);
                Self::extract_service_patterns(right, subqueries);
            }
            Algebra::LeftJoin { left, right, .. } => {
                Self::extract_service_patterns(left, subqueries);
                Self::extract_service_patterns(right, subqueries);
            }
            Algebra::Filter { pattern, .. }
            | Algebra::Extend { pattern, .. }
            | Algebra::Graph { pattern, .. }
            | Algebra::Project { pattern, .. }
            | Algebra::Distinct { pattern }
            | Algebra::Reduced { pattern }
            | Algebra::Slice { pattern, .. }
            | Algebra::OrderBy { pattern, .. }
            | Algebra::Group { pattern, .. }
            | Algebra::Having { pattern, .. } => {
                Self::extract_service_patterns(pattern, subqueries);
            }
            _ => {}
        }
    }

    /// Execute federated query with parallel execution
    pub async fn execute_federated_query(
        &self,
        subqueries: Vec<FederatedSubquery>,
    ) -> Result<Solution> {
        let mut tasks = Vec::new();

        for subquery in subqueries {
            let executor = self.clone();
            let task = tokio::spawn(async move {
                executor
                    .execute_service(&subquery.endpoint, &subquery.pattern, subquery.silent)
                    .await
            });
            tasks.push(task);
        }

        // Collect results
        let mut all_results = Vec::new();
        for task in tasks {
            match task.await {
                Ok(Ok(results)) => all_results.push(results),
                Ok(Err(e)) => tracing::warn!("Federated subquery failed: {}", e),
                Err(e) => tracing::error!("Task join error: {}", e),
            }
        }

        // Merge all results
        Ok(self.merge_results(all_results))
    }

    /// Merge results from multiple endpoints
    pub fn merge_results(&self, results: Vec<Solution>) -> Solution {
        // Simple concatenation with duplicate elimination
        // Since Binding (HashMap<Variable, Term>) implements PartialEq, we can eliminate duplicates
        let mut merged: Solution = Vec::new();

        for result_set in results {
            for binding in result_set {
                // Check if binding already exists
                if !merged.iter().any(|b| b == &binding) {
                    merged.push(binding);
                }
            }
        }

        merged
    }

    /// Get federation statistics
    pub fn statistics(&self) -> FederationStats {
        let stats = self.metrics.request_duration.get_stats();
        FederationStats {
            total_requests: self.metrics.successful_requests.get()
                + self.metrics.failed_requests.get(),
            successful_requests: self.metrics.successful_requests.get(),
            failed_requests: self.metrics.failed_requests.get(),
            cache_hits: self.metrics.cache_hits.get(),
            cache_misses: self.metrics.cache_misses.get(),
            average_request_duration: stats.mean,
            total_results: self.metrics.results_received.get(),
            active_endpoints: self.endpoint_health.len(),
            healthy_endpoints: self
                .endpoint_health
                .iter()
                .filter(|e| e.value().is_healthy())
                .count(),
        }
    }
}

impl Clone for FederationExecutor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            connection_pools: Arc::clone(&self.connection_pools),
            result_cache: Arc::clone(&self.result_cache),
            endpoint_health: Arc::clone(&self.endpoint_health),
            metrics: Arc::clone(&self.metrics),
            request_semaphore: Arc::clone(&self.request_semaphore),
        }
    }
}

/// Connection pool for a SPARQL endpoint
pub struct ConnectionPool {
    endpoint: String,
    size: usize,
    active_connections: parking_lot::Mutex<usize>,
}

impl ConnectionPool {
    fn new(endpoint: String, size: usize) -> Self {
        Self {
            endpoint,
            size,
            active_connections: parking_lot::Mutex::new(0),
        }
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn available(&self) -> usize {
        let active = *self.active_connections.lock();
        self.size.saturating_sub(active)
    }
}

/// Cache key for query results
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct QueryCacheKey {
    endpoint: String,
    query: String,
}

/// Cached query result
struct CachedResult {
    results: Solution,
    timestamp: Instant,
    ttl: Duration,
}

impl CachedResult {
    fn is_expired(&self) -> bool {
        self.timestamp.elapsed() > self.ttl
    }
}

/// Endpoint health status
#[derive(Debug, Clone)]
pub struct EndpointHealth {
    /// Is endpoint healthy
    healthy: bool,
    /// Last successful request time
    last_success: Option<Instant>,
    /// Last failure time
    last_failure: Option<Instant>,
    /// Consecutive failures
    consecutive_failures: usize,
    /// Average response time
    avg_response_time: Duration,
    /// Status message
    status_message: String,
}

impl EndpointHealth {
    fn new() -> Self {
        Self {
            healthy: true,
            last_success: None,
            last_failure: None,
            consecutive_failures: 0,
            avg_response_time: Duration::from_secs(0),
            status_message: "OK".to_string(),
        }
    }

    fn is_healthy(&self) -> bool {
        self.healthy
    }

    fn record_success(&mut self, response_time: Duration) {
        self.healthy = true;
        self.last_success = Some(Instant::now());
        self.consecutive_failures = 0;
        self.avg_response_time = response_time;
        self.status_message = "OK".to_string();
    }

    fn record_failure(&mut self) {
        self.last_failure = Some(Instant::now());
        self.consecutive_failures += 1;

        // Mark unhealthy after 3 consecutive failures
        if self.consecutive_failures >= 3 {
            self.healthy = false;
            self.status_message = format!("{} consecutive failures", self.consecutive_failures);
        }
    }
}

/// Federated subquery
#[derive(Debug, Clone)]
pub struct FederatedSubquery {
    pub endpoint: Term,
    pub pattern: Algebra,
    pub silent: bool,
    pub dependencies: Vec<Variable>,
}

/// Federation metrics
struct FederationMetrics {
    successful_requests: Counter,
    failed_requests: Counter,
    retried_requests: Counter,
    cache_hits: Counter,
    cache_misses: Counter,
    request_duration: Timer,
    results_received: Counter,
}

impl FederationMetrics {
    fn new() -> Self {
        Self {
            successful_requests: Counter::new("federation.successful_requests".to_string()),
            failed_requests: Counter::new("federation.failed_requests".to_string()),
            retried_requests: Counter::new("federation.retried_requests".to_string()),
            cache_hits: Counter::new("federation.cache_hits".to_string()),
            cache_misses: Counter::new("federation.cache_misses".to_string()),
            request_duration: Timer::new("federation.request_duration".to_string()),
            results_received: Counter::new("federation.results_received".to_string()),
        }
    }
}

/// Federation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationStats {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub average_request_duration: f64,
    pub total_results: u64,
    pub active_endpoints: usize,
    pub healthy_endpoints: usize,
}

/// Endpoint capability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointCapabilities {
    /// Endpoint URL
    pub endpoint: String,
    /// Supported SPARQL version
    pub sparql_version: String,
    /// Supported result formats
    pub result_formats: Vec<String>,
    /// Maximum query complexity
    pub max_query_complexity: Option<usize>,
    /// Supports federation
    pub supports_federation: bool,
    /// Supports RDF-star
    pub supports_rdf_star: bool,
    /// Available named graphs
    pub named_graphs: Vec<String>,
}

/// Endpoint discovery service
pub struct EndpointDiscovery {
    /// Known endpoints
    endpoints: Arc<DashMap<String, EndpointCapabilities>>,
}

impl EndpointDiscovery {
    /// Create a new endpoint discovery service
    pub fn new() -> Self {
        Self {
            endpoints: Arc::new(DashMap::new()),
        }
    }

    /// Register an endpoint
    pub fn register_endpoint(&self, capabilities: EndpointCapabilities) {
        self.endpoints
            .insert(capabilities.endpoint.clone(), capabilities);
    }

    /// Discover endpoint capabilities using SPARQL Service Description
    pub async fn discover_endpoint(&self, endpoint: &str) -> Result<EndpointCapabilities> {
        // Query the endpoint for service description
        // Using SPARQL Service Description vocabulary from W3C
        let service_description_query = r#"
            PREFIX sd: <http://www.w3.org/ns/sparql-service-description#>
            PREFIX void: <http://rdfs.org/ns/void#>

            SELECT ?feature ?format ?version ?graph WHERE {
                {
                    ?service a sd:Service .
                    OPTIONAL { ?service sd:feature ?feature }
                    OPTIONAL { ?service sd:resultFormat ?format }
                    OPTIONAL { ?service sd:languageExtension ?version }
                    OPTIONAL { ?service sd:defaultDataset/sd:namedGraph/sd:name ?graph }
                }
            }
        "#;

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;

        // Try to query for service description
        let response_result = client
            .post(endpoint)
            .header("Accept", "application/sparql-results+json")
            .header("Content-Type", "application/sparql-query")
            .body(service_description_query)
            .send()
            .await;

        let mut capabilities = EndpointCapabilities {
            endpoint: endpoint.to_string(),
            sparql_version: "1.1".to_string(),
            result_formats: vec!["application/sparql-results+json".to_string()],
            max_query_complexity: None,
            supports_federation: false,
            supports_rdf_star: false,
            named_graphs: Vec::new(),
        };

        match response_result {
            Ok(response) if response.status().is_success() => {
                // Parse service description response
                if let Ok(json) = response.json::<serde_json::Value>().await {
                    if let Some(bindings) = json["results"]["bindings"].as_array() {
                        let mut formats = Vec::new();
                        let mut graphs = Vec::new();

                        for binding in bindings {
                            // Extract result formats
                            if let Some(format) = binding["format"]["value"].as_str() {
                                formats.push(format.to_string());
                            }

                            // Extract SPARQL version/features
                            if let Some(feature) = binding["feature"]["value"].as_str() {
                                if feature.contains("UnionDefaultGraph") {
                                    capabilities.supports_federation = true;
                                }
                                if feature.contains("SPARQL-star") || feature.contains("RDFstar") {
                                    capabilities.supports_rdf_star = true;
                                }
                                if feature.contains("1.2") {
                                    capabilities.sparql_version = "1.2".to_string();
                                }
                            }

                            // Extract named graphs
                            if let Some(graph) = binding["graph"]["value"].as_str() {
                                graphs.push(graph.to_string());
                            }
                        }

                        if !formats.is_empty() {
                            capabilities.result_formats = formats;
                        }
                        capabilities.named_graphs = graphs;
                    }
                }

                tracing::info!(
                    "Discovered endpoint capabilities for {}: version={}, formats={:?}, federation={}, rdf-star={}",
                    endpoint,
                    capabilities.sparql_version,
                    capabilities.result_formats,
                    capabilities.supports_federation,
                    capabilities.supports_rdf_star
                );
            }
            Ok(response) => {
                tracing::warn!(
                    "Service description query failed with status {}: {}",
                    response.status(),
                    response.text().await.unwrap_or_default()
                );
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to query service description from {}: {}. Using default capabilities.",
                    endpoint,
                    e
                );
            }
        }

        // Try a simple ASK query to verify endpoint is responsive
        let test_query = "ASK { ?s ?p ?o }";
        if let Ok(response) = client
            .post(endpoint)
            .header("Accept", "application/sparql-results+json")
            .header("Content-Type", "application/sparql-query")
            .body(test_query)
            .send()
            .await
        {
            if response.status().is_success() {
                tracing::debug!("Endpoint {} is responsive", endpoint);
            }
        }

        Ok(capabilities)
    }

    /// Find endpoints matching criteria
    pub fn find_endpoints(&self, criteria: EndpointCriteria) -> Vec<EndpointCapabilities> {
        self.endpoints
            .iter()
            .filter(|entry| {
                let caps = entry.value();
                (criteria.supports_federation.is_none()
                    || criteria.supports_federation == Some(caps.supports_federation))
                    && (criteria.supports_rdf_star.is_none()
                        || criteria.supports_rdf_star == Some(caps.supports_rdf_star))
            })
            .map(|entry| entry.value().clone())
            .collect()
    }
}

impl Default for EndpointDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

/// Endpoint search criteria
#[derive(Debug, Clone, Default)]
pub struct EndpointCriteria {
    pub supports_federation: Option<bool>,
    pub supports_rdf_star: Option<bool>,
    pub min_sparql_version: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::TriplePattern;

    #[test]
    fn test_federation_config_default() {
        let config = FederationConfig::default();
        assert_eq!(config.max_concurrent_requests, 32);
        assert_eq!(config.max_retries, 3);
        assert!(config.enable_caching);
    }

    #[test]
    fn test_endpoint_health() {
        let mut health = EndpointHealth::new();
        assert!(health.is_healthy());

        health.record_failure();
        assert!(health.is_healthy());

        health.record_failure();
        health.record_failure();
        assert!(!health.is_healthy());

        health.record_success(Duration::from_millis(100));
        assert!(health.is_healthy());
    }

    #[test]
    fn test_endpoint_discovery() {
        let discovery = EndpointDiscovery::new();

        let caps = EndpointCapabilities {
            endpoint: "http://example.org/sparql".to_string(),
            sparql_version: "1.1".to_string(),
            result_formats: vec!["application/sparql-results+json".to_string()],
            max_query_complexity: None,
            supports_federation: true,
            supports_rdf_star: false,
            named_graphs: Vec::new(),
        };

        discovery.register_endpoint(caps.clone());

        let found = discovery.find_endpoints(EndpointCriteria {
            supports_federation: Some(true),
            ..Default::default()
        });

        assert_eq!(found.len(), 1);
        assert_eq!(found[0].endpoint, caps.endpoint);
    }

    #[tokio::test]
    async fn test_federation_executor() {
        let config = FederationConfig::default();
        let executor = FederationExecutor::new(config);

        let stats = executor.statistics();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.active_endpoints, 0);
    }

    #[test]
    fn test_algebra_to_sparql_bgp() {
        let executor = FederationExecutor::new(FederationConfig::default());

        // Create a simple BGP
        let patterns = vec![TriplePattern {
            subject: Term::Variable(Variable::new_unchecked("s")),
            predicate: Term::Iri(crate::algebra::Iri::new_unchecked(
                "http://example.org/predicate",
            )),
            object: Term::Variable(Variable::new_unchecked("o")),
        }];

        let algebra = Algebra::Bgp(patterns);
        let sparql = executor.algebra_to_sparql(&algebra).unwrap();

        assert!(sparql.contains("SELECT * WHERE"));
        assert!(sparql.contains("?s"));
        assert!(sparql.contains("<http://example.org/predicate>"));
        assert!(sparql.contains("?o"));
    }

    #[test]
    fn test_parse_sparql_json_results() {
        let executor = FederationExecutor::new(FederationConfig::default());

        let json_str = r#"{
            "results": {
                "bindings": [
                    {
                        "s": {
                            "type": "uri",
                            "value": "http://example.org/subject1"
                        },
                        "o": {
                            "type": "literal",
                            "value": "object1"
                        }
                    }
                ]
            }
        }"#;

        let json: serde_json::Value = serde_json::from_str(json_str).unwrap();
        let results = executor.parse_sparql_json_results(&json).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 2); // s, o
    }

    #[test]
    fn test_parse_sparql_json_term_types() {
        let executor = FederationExecutor::new(FederationConfig::default());

        // Test URI
        let uri_json = serde_json::json!({
            "type": "uri",
            "value": "http://example.org/resource"
        });
        let term = executor.parse_sparql_json_term(&uri_json).unwrap();
        matches!(term, Term::Iri(_));

        // Test literal
        let literal_json = serde_json::json!({
            "type": "literal",
            "value": "test value"
        });
        let term = executor.parse_sparql_json_term(&literal_json).unwrap();
        match term {
            Term::Literal(lit) => assert_eq!(lit.value, "test value"),
            _ => panic!("Expected Literal term"),
        }

        // Test blank node
        let bnode_json = serde_json::json!({
            "type": "bnode",
            "value": "b0"
        });
        let term = executor.parse_sparql_json_term(&bnode_json).unwrap();
        match term {
            Term::BlankNode(id) => assert_eq!(id, "b0"),
            _ => panic!("Expected BlankNode term"),
        }
    }

    #[test]
    fn test_merge_results() {
        let executor = FederationExecutor::new(FederationConfig::default());

        let var_s = Variable::new_unchecked("s");

        let mut binding1 = Binding::new();
        binding1.insert(
            var_s.clone(),
            Term::Iri(crate::algebra::Iri::new_unchecked("http://example.org/s1")),
        );

        let mut binding2 = Binding::new();
        binding2.insert(
            var_s.clone(),
            Term::Iri(crate::algebra::Iri::new_unchecked("http://example.org/s2")),
        );

        let solution1 = vec![binding1.clone()];
        let solution2 = vec![binding2, binding1.clone()]; // Contains duplicate

        let merged = executor.merge_results(vec![solution1, solution2]);

        // Should eliminate duplicates
        assert_eq!(merged.len(), 2); // Only 2 unique bindings
    }
}
