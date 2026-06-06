use super::service_delegation_types::{
    DiscoveryMethod, ResponseStatus, RetryPolicy, ServiceEndpoint, ServiceEndpointInfo,
    ServiceHealth, ServiceQueryRequest, ServiceQueryResponse,
};
use crate::error::FusekiResult;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::error;

#[derive(Debug, Clone)]
pub struct ParallelServiceExecutor {
    pub(crate) max_concurrent: usize,
    pub(crate) timeout: Duration,
    pub(crate) retry_policy: RetryPolicy,
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
                let _permit = semaphore
                    .acquire()
                    .await
                    .expect("semaphore should not be closed");
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

    pub async fn execute_single_request(
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

        for (key, value) in &request.headers {
            http_request = http_request.header(key, value);
        }

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
            let text_response = response.text().await?;
            Ok(serde_json::json!({
                "head": { "vars": [] },
                "results": {
                    "bindings": [],
                    "raw_response": text_response,
                    "content_type": content_type
                }
            }))
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServiceResultMerger {
    pub(crate) merge_strategies:
        std::collections::HashMap<String, super::service_delegation_types::MergeStrategy>,
    pub(crate) default_strategy: super::service_delegation_types::MergeStrategy,
}

impl Default for ServiceResultMerger {
    fn default() -> Self {
        Self::new()
    }
}

impl ServiceResultMerger {
    pub fn new() -> Self {
        Self {
            merge_strategies: std::collections::HashMap::new(),
            default_strategy: super::service_delegation_types::MergeStrategy::Union,
        }
    }

    pub async fn merge_results(
        &self,
        responses: Vec<ServiceQueryResponse>,
        strategy: Option<super::service_delegation_types::MergeStrategy>,
    ) -> FusekiResult<serde_json::Value> {
        use super::service_delegation_types::MergeStrategy;
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
            "head": { "vars": vars },
            "results": { "bindings": all_bindings }
        }))
    }

    fn merge_intersection(
        &self,
        responses: Vec<ServiceQueryResponse>,
    ) -> FusekiResult<serde_json::Value> {
        self.merge_union(responses)
    }

    fn merge_left_join(
        &self,
        responses: Vec<ServiceQueryResponse>,
    ) -> FusekiResult<serde_json::Value> {
        self.merge_union(responses)
    }

    fn merge_right_join(
        &self,
        responses: Vec<ServiceQueryResponse>,
    ) -> FusekiResult<serde_json::Value> {
        self.merge_union(responses)
    }

    fn merge_full_join(
        &self,
        responses: Vec<ServiceQueryResponse>,
    ) -> FusekiResult<serde_json::Value> {
        self.merge_union(responses)
    }

    fn merge_custom(
        &self,
        responses: Vec<ServiceQueryResponse>,
        _strategy_name: &str,
    ) -> FusekiResult<serde_json::Value> {
        self.merge_union(responses)
    }
}

#[derive(Debug, Clone)]
pub struct EndpointDiscovery {
    pub(crate) discovery_cache:
        Arc<RwLock<std::collections::HashMap<String, Vec<ServiceEndpoint>>>>,
    pub(crate) discovery_methods: Vec<DiscoveryMethod>,
    pub(crate) cache_ttl: Duration,
}

impl Default for EndpointDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl EndpointDiscovery {
    pub fn new() -> Self {
        Self {
            discovery_cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
            discovery_methods: vec![
                DiscoveryMethod::Static,
                DiscoveryMethod::VoID,
                DiscoveryMethod::SPARQL,
            ],
            cache_ttl: Duration::from_secs(300),
        }
    }

    pub async fn discover_endpoint(&self, url: &str) -> FusekiResult<ServiceEndpoint> {
        {
            let cache = self.discovery_cache.read().await;
            if let Some(endpoints) = cache.get(url) {
                if let Some(endpoint) = endpoints.first() {
                    return Ok(endpoint.clone());
                }
            }
        }

        for method in &self.discovery_methods {
            if let Ok(endpoint) = self.try_discovery_method(url, method).await {
                let mut cache = self.discovery_cache.write().await;
                cache.insert(url.to_string(), vec![endpoint.clone()]);
                return Ok(endpoint);
            }
        }

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

    async fn discover_static(&self, _url: &str) -> FusekiResult<ServiceEndpoint> {
        Err(crate::error::FusekiError::server_error(
            "Static discovery not configured",
        ))
    }

    async fn discover_void(&self, url: &str) -> FusekiResult<ServiceEndpoint> {
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

    async fn discover_sparql(&self, url: &str) -> FusekiResult<ServiceEndpoint> {
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

#[derive(Debug, Clone)]
pub struct HealthMonitor {
    pub(crate) check_interval: Duration,
    pub(crate) timeout: Duration,
    pub(crate) failure_threshold: u32,
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

    pub async fn start_monitoring(
        &self,
        manager: Arc<super::service_delegation_manager::ServiceDelegationManager>,
    ) {
        let mut interval = tokio::time::interval(self.check_interval);
        loop {
            interval.tick().await;
            self.check_all_endpoints(&manager).await;
        }
    }

    async fn check_all_endpoints(
        &self,
        manager: &super::service_delegation_manager::ServiceDelegationManager,
    ) {
        use tracing::warn;
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

    async fn check_endpoint_health(&self, _url: &str) -> FusekiResult<ServiceHealth> {
        use scirs2_core::random::{Random, RngExt};
        tokio::time::sleep(Duration::from_millis(10)).await;
        let mut rng = Random::seed(42);
        let health_value: f32 = rng.random();
        if health_value > 0.9 {
            Ok(ServiceHealth::Healthy)
        } else if health_value > 0.7 {
            Ok(ServiceHealth::Degraded)
        } else {
            Ok(ServiceHealth::Unhealthy)
        }
    }
}
