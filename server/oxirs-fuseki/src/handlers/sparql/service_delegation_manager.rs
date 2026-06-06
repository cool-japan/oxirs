use super::service_delegation_executor::{
    EndpointDiscovery, HealthMonitor, ParallelServiceExecutor, ServiceResultMerger,
};
use super::service_delegation_types::{
    CacheStats, EndpointStats, LoadBalancer, LoadBalancingStrategy, MergeStrategy, QueryCache,
    ResponseStatus, RetryPolicy, ServiceEndpoint, ServiceEndpointInfo, ServiceHealth,
    ServiceQueryRequest, ServiceQueryResponse,
};
use crate::error::FusekiResult;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::debug;

#[derive(Debug, Clone)]
pub struct ServiceDelegationManager {
    pub(crate) endpoints: Arc<RwLock<HashMap<String, ServiceEndpoint>>>,
    executor: ParallelServiceExecutor,
    merger: ServiceResultMerger,
    discovery: EndpointDiscovery,
    health_monitor: HealthMonitor,
    query_cache: Arc<RwLock<QueryCache>>,
    load_balancer: LoadBalancer,
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

    pub async fn register_endpoint(&self, endpoint: ServiceEndpoint) -> FusekiResult<()> {
        let mut endpoints = self.endpoints.write().await;
        endpoints.insert(endpoint.url.clone(), endpoint);
        Ok(())
    }

    pub async fn process_service_clauses(&self, query: &str) -> FusekiResult<String> {
        let service_clauses = self.extract_service_clauses(query)?;
        let mut processed_query = query.to_string();
        for service_clause in service_clauses {
            let optimized_clause = self.optimize_service_clause(&service_clause).await?;
            processed_query = processed_query.replace(&service_clause, &optimized_clause);
        }
        Ok(processed_query)
    }

    pub(crate) fn extract_service_clauses(&self, query: &str) -> FusekiResult<Vec<String>> {
        let mut clauses = Vec::new();
        let mut pos = 0;
        while let Some(service_pos) = query[pos..].find("SERVICE") {
            let abs_pos = pos + service_pos;
            if let Some(clause) = self.extract_complete_service_clause(&query[abs_pos..]) {
                let clause_len = clause.len();
                clauses.push(clause);
                pos = abs_pos + clause_len;
            } else {
                pos = abs_pos + 7;
            }
        }
        Ok(clauses)
    }

    fn extract_complete_service_clause(&self, text: &str) -> Option<String> {
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

    async fn optimize_service_clause(&self, service_clause: &str) -> FusekiResult<String> {
        let (service_url, inner_query) = self.parse_service_clause(service_clause)?;
        let endpoint = self.get_or_discover_endpoint(&service_url).await?;
        let optimized_inner = self.optimize_for_endpoint(&inner_query, &endpoint).await?;
        Ok(format!("SERVICE <{service_url}> {{ {optimized_inner} }}"))
    }

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

    async fn get_or_discover_endpoint(&self, url: &str) -> FusekiResult<ServiceEndpoint> {
        {
            let endpoints = self.endpoints.read().await;
            if let Some(endpoint) = endpoints.get(url) {
                return Ok(endpoint.clone());
            }
        }
        let discovered = self.discovery.discover_endpoint(url).await?;
        self.register_endpoint(discovered.clone()).await?;
        Ok(discovered)
    }

    async fn optimize_for_endpoint(
        &self,
        query: &str,
        endpoint: &ServiceEndpoint,
    ) -> FusekiResult<String> {
        let mut optimized = query.to_string();
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

    fn apply_sparql_11_optimizations(&self, query: &str) -> FusekiResult<String> {
        Ok(query.to_string())
    }

    fn optimize_bind_for_endpoint(&self, query: &str) -> FusekiResult<String> {
        Ok(query.to_string())
    }

    fn optimize_values_for_endpoint(&self, query: &str) -> FusekiResult<String> {
        Ok(query.to_string())
    }

    pub async fn execute_federated_query(
        &self,
        requests: Vec<ServiceQueryRequest>,
    ) -> FusekiResult<Vec<ServiceQueryResponse>> {
        self.executor.execute_parallel(requests).await
    }

    pub async fn merge_service_results(
        &self,
        responses: Vec<ServiceQueryResponse>,
        strategy: Option<MergeStrategy>,
    ) -> FusekiResult<serde_json::Value> {
        self.merger.merge_results(responses, strategy).await
    }

    pub async fn get_endpoint_health(&self, url: &str) -> Option<ServiceHealth> {
        let endpoints = self.endpoints.read().await;
        endpoints.get(url).map(|e| e.health_status.clone())
    }

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

    pub async fn execute_service_query_optimized(
        &self,
        service_url: &str,
        query: &str,
        use_cache: bool,
    ) -> FusekiResult<ServiceQueryResponse> {
        let cache_key = format!("{service_url}:{query}");

        if use_cache {
            let mut cache = self.query_cache.write().await;
            if let Some(cached_result) = cache.get(&cache_key) {
                debug!("Cache hit for SERVICE query: {}", service_url);
                return Ok(ServiceQueryResponse {
                    status: ResponseStatus::Success,
                    results: Some(cached_result),
                    error_message: None,
                    execution_time: Duration::from_millis(1),
                    endpoint_info: ServiceEndpointInfo {
                        url: service_url.to_string(),
                        response_time: Duration::from_millis(1),
                        attempt_count: 1,
                    },
                });
            }
        }

        let endpoints = self.get_available_endpoints(service_url).await?;

        let selected_endpoint = if endpoints.len() > 1 {
            self.load_balancer.select_endpoint(&endpoints)
        } else {
            endpoints.first()
        };

        let endpoint = selected_endpoint
            .ok_or_else(|| crate::error::FusekiError::bad_request("No available endpoints"))?;

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

        if use_cache && response.status == ResponseStatus::Success {
            if let Some(ref result) = response.results {
                let mut cache = self.query_cache.write().await;
                cache.put(cache_key, result.clone(), None);
                debug!("Cached SERVICE query result for: {}", service_url);
            }
        }

        Ok(response)
    }

    async fn get_available_endpoints(
        &self,
        service_url: &str,
    ) -> FusekiResult<Vec<ServiceEndpoint>> {
        let endpoints = self.endpoints.read().await;
        if let Some(endpoint) = endpoints.get(service_url) {
            if endpoint.health_status != ServiceHealth::Unhealthy {
                return Ok(vec![endpoint.clone()]);
            }
        }
        let discovered = self.discovery.discover_endpoint(service_url).await?;
        Ok(vec![discovered])
    }

    pub async fn update_load_balancer_metrics(
        &mut self,
        url: &str,
        response_time: Duration,
        success: bool,
    ) {
        let health_score = if success { 1.0 } else { 0.0 };
        self.load_balancer
            .set_endpoint_health_score(url.to_string(), health_score);
        let weight = if response_time.as_millis() > 0 {
            1000.0 / response_time.as_millis() as f64
        } else {
            10.0
        };
        self.load_balancer
            .set_endpoint_weight(url.to_string(), weight);
    }

    pub async fn get_cache_stats(&self) -> CacheStats {
        let cache = self.query_cache.read().await;
        cache.stats()
    }

    pub fn get_endpoint_stats(&self, url: &str) -> EndpointStats {
        self.load_balancer.get_endpoint_stats(url)
    }

    pub async fn cleanup_cache(&self) {
        let mut cache = self.query_cache.write().await;
        cache.cleanup_expired();
    }
}
