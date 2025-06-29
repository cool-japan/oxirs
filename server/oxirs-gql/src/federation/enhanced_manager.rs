//! Enhanced Federation Manager with Dynamic Service Discovery
//!
//! This module provides an advanced federation manager that combines service discovery,
//! intelligent query routing, load balancing, and fault tolerance.

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex};
use tracing::{debug, error, info, warn};

use super::config::{FederationConfig, RemoteEndpoint, RetryStrategy};
use super::query_planner::{QueryPlan, QueryPlanner};
use super::schema_stitcher::SchemaStitcher;
use super::service_discovery::{
    ServiceDiscovery, ServiceDiscoveryConfig, DiscoveryEvent,
    ServiceDiscoveryEventHandler, ServiceInfo, HealthStatus
};
use crate::ast::{Document, Value, OperationType};
use crate::types::Schema;

/// Enhanced federation manager configuration
#[derive(Debug, Clone)]
pub struct EnhancedFederationConfig {
    pub service_discovery: ServiceDiscoveryConfig,
    pub load_balancing: LoadBalancingConfig,
    pub circuit_breaker: CircuitBreakerConfig,
    pub retry_policy: RetryPolicyConfig,
    pub query_routing: QueryRoutingConfig,
    pub caching: FederationCacheConfig,
}

impl Default for EnhancedFederationConfig {
    fn default() -> Self {
        Self {
            service_discovery: ServiceDiscoveryConfig::default(),
            load_balancing: LoadBalancingConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            retry_policy: RetryPolicyConfig::default(),
            query_routing: QueryRoutingConfig::default(),
            caching: FederationCacheConfig::default(),
        }
    }
}

/// Load balancing configuration
#[derive(Debug, Clone)]
pub struct LoadBalancingConfig {
    pub algorithm: LoadBalancingAlgorithm,
    pub health_check_weight: f64,
    pub response_time_weight: f64,
    pub load_weight: f64,
    pub max_requests_per_service: usize,
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            algorithm: LoadBalancingAlgorithm::WeightedRoundRobin,
            health_check_weight: 0.4,
            response_time_weight: 0.3,
            load_weight: 0.3,
            max_requests_per_service: 1000,
        }
    }
}

/// Load balancing algorithms
#[derive(Debug, Clone)]
pub enum LoadBalancingAlgorithm {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    LeastResponseTime,
    Adaptive,
    ConsistentHashing,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: usize,
    pub success_threshold: usize,
    pub timeout: Duration,
    pub retry_after: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(30),
            retry_after: Duration::from_secs(60),
        }
    }
}

/// Retry policy configuration
#[derive(Debug, Clone)]
pub struct RetryPolicyConfig {
    pub max_retries: usize,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
    pub jitter: bool,
}

impl Default for RetryPolicyConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

/// Query routing configuration
#[derive(Debug, Clone)]
pub struct QueryRoutingConfig {
    pub enable_query_analysis: bool,
    pub enable_field_based_routing: bool,
    pub enable_cost_based_routing: bool,
    pub parallel_execution: bool,
    pub max_parallel_requests: usize,
}

impl Default for QueryRoutingConfig {
    fn default() -> Self {
        Self {
            enable_query_analysis: true,
            enable_field_based_routing: true,
            enable_cost_based_routing: true,
            parallel_execution: true,
            max_parallel_requests: 10,
        }
    }
}

/// Federation cache configuration
#[derive(Debug, Clone)]
pub struct FederationCacheConfig {
    pub enable_schema_caching: bool,
    pub enable_query_caching: bool,
    pub schema_cache_ttl: Duration,
    pub query_cache_ttl: Duration,
    pub max_cache_size: usize,
}

impl Default for FederationCacheConfig {
    fn default() -> Self {
        Self {
            enable_schema_caching: true,
            enable_query_caching: true,
            schema_cache_ttl: Duration::from_secs(3600),
            query_cache_ttl: Duration::from_secs(300),
            max_cache_size: 1000,
        }
    }
}

/// Circuit breaker state
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

/// Service circuit breaker
#[derive(Debug)]
pub struct ServiceCircuitBreaker {
    state: CircuitBreakerState,
    failure_count: usize,
    success_count: usize,
    last_failure_time: Option<Instant>,
    config: CircuitBreakerConfig,
}

impl ServiceCircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: CircuitBreakerState::Closed,
            failure_count: 0,
            success_count: 0,
            last_failure_time: None,
            config,
        }
    }

    pub fn can_execute(&mut self) -> bool {
        match self.state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() >= self.config.retry_after {
                        self.state = CircuitBreakerState::HalfOpen;
                        self.success_count = 0;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitBreakerState::HalfOpen => true,
        }
    }

    pub fn record_success(&mut self) {
        match self.state {
            CircuitBreakerState::Closed => {
                self.failure_count = 0;
            }
            CircuitBreakerState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.config.success_threshold {
                    self.state = CircuitBreakerState::Closed;
                    self.failure_count = 0;
                }
            }
            CircuitBreakerState::Open => {}
        }
    }

    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(Instant::now());

        match self.state {
            CircuitBreakerState::Closed => {
                if self.failure_count >= self.config.failure_threshold {
                    self.state = CircuitBreakerState::Open;
                }
            }
            CircuitBreakerState::HalfOpen => {
                self.state = CircuitBreakerState::Open;
            }
            CircuitBreakerState::Open => {}
        }
    }

    pub fn state(&self) -> CircuitBreakerState {
        self.state.clone()
    }
}

/// Federation execution context
#[derive(Debug, Clone)]
pub struct FederationContext {
    pub request_id: String,
    pub user_id: Option<String>,
    pub headers: HashMap<String, String>,
    pub variables: HashMap<String, Value>,
    pub operation_name: Option<String>,
}

/// Query execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationResult {
    pub data: Option<serde_json::Value>,
    pub errors: Vec<FederationError>,
    pub extensions: HashMap<String, serde_json::Value>,
}

/// Federation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationError {
    pub message: String,
    pub path: Option<Vec<String>>,
    pub locations: Option<Vec<ErrorLocation>>,
    pub extensions: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorLocation {
    pub line: usize,
    pub column: usize,
}

/// Enhanced federation manager
pub struct EnhancedFederationManager {
    config: EnhancedFederationConfig,
    service_discovery: Arc<ServiceDiscovery>,
    schema_stitcher: Arc<SchemaStitcher>,
    query_planner: Arc<QueryPlanner>,
    circuit_breakers: Arc<RwLock<HashMap<String, ServiceCircuitBreaker>>>,
    load_balancer: Arc<Mutex<LoadBalancer>>,
    merged_schema: Arc<RwLock<Option<Schema>>>,
    http_client: reqwest::Client,
}

/// Load balancer implementation
#[derive(Debug)]
pub struct LoadBalancer {
    algorithm: LoadBalancingAlgorithm,
    round_robin_counter: usize,
    request_counts: HashMap<String, usize>,
    config: LoadBalancingConfig,
}

impl LoadBalancer {
    pub fn new(config: LoadBalancingConfig) -> Self {
        Self {
            algorithm: config.algorithm.clone(),
            round_robin_counter: 0,
            request_counts: HashMap::new(),
            config,
        }
    }

    /// Select the best service for a request
    pub fn select_service(&mut self, services: &[ServiceInfo], _query_hash: Option<u64>) -> Option<String> {
        if services.is_empty() {
            return None;
        }

        let healthy_services: Vec<_> = services
            .iter()
            .filter(|s| s.health_status == HealthStatus::Healthy)
            .collect();

        if healthy_services.is_empty() {
            return None;
        }

        match self.algorithm {
            LoadBalancingAlgorithm::RoundRobin => {
                let service = &healthy_services[self.round_robin_counter % healthy_services.len()];
                self.round_robin_counter += 1;
                Some(service.id.clone())
            }
            LoadBalancingAlgorithm::WeightedRoundRobin => {
                self.select_weighted_round_robin(&healthy_services)
            }
            LoadBalancingAlgorithm::LeastConnections => {
                self.select_least_connections(&healthy_services)
            }
            LoadBalancingAlgorithm::LeastResponseTime => {
                self.select_least_response_time(&healthy_services)
            }
            LoadBalancingAlgorithm::Adaptive => {
                self.select_adaptive(&healthy_services)
            }
            LoadBalancingAlgorithm::ConsistentHashing => {
                // For consistent hashing, we'd use the query_hash
                self.select_consistent_hash(&healthy_services, _query_hash.unwrap_or(0))
            }
        }
    }

    fn select_weighted_round_robin(&mut self, services: &[&ServiceInfo]) -> Option<String> {
        // Calculate weights based on health, response time, and load
        let mut best_service = None;
        let mut best_score = f64::INFINITY;

        for service in services {
            let health_score = match service.health_status {
                HealthStatus::Healthy => 1.0,
                HealthStatus::Degraded => 0.5,
                _ => 0.0,
            };

            let response_time_score = service.response_time
                .map(|rt| rt.as_millis() as f64)
                .unwrap_or(1000.0);

            let load_score = service.load_factor;
            let request_count = self.request_counts.get(&service.id).copied().unwrap_or(0) as f64;

            let total_score = 
                (health_score * self.config.health_check_weight) +
                (response_time_score * self.config.response_time_weight) +
                (load_score * self.config.load_weight) +
                (request_count * 0.1);

            if total_score < best_score {
                best_score = total_score;
                best_service = Some(service.id.clone());
            }
        }

        if let Some(ref service_id) = best_service {
            *self.request_counts.entry(service_id.clone()).or_insert(0) += 1;
        }

        best_service
    }

    fn select_least_connections(&self, services: &[&ServiceInfo]) -> Option<String> {
        services
            .iter()
            .min_by_key(|service| {
                self.request_counts.get(&service.id).copied().unwrap_or(0)
            })
            .map(|service| service.id.clone())
    }

    fn select_least_response_time(&self, services: &[&ServiceInfo]) -> Option<String> {
        services
            .iter()
            .min_by_key(|service| {
                service.response_time.unwrap_or(Duration::from_secs(10))
            })
            .map(|service| service.id.clone())
    }

    fn select_adaptive(&mut self, services: &[&ServiceInfo]) -> Option<String> {
        // Adaptive algorithm combines multiple factors
        self.select_weighted_round_robin(services)
    }

    fn select_consistent_hash(&self, services: &[&ServiceInfo], hash: u64) -> Option<String> {
        if services.is_empty() {
            return None;
        }

        let index = (hash as usize) % services.len();
        Some(services[index].id.clone())
    }

    pub fn record_completion(&mut self, service_id: &str) {
        if let Some(count) = self.request_counts.get_mut(service_id) {
            if *count > 0 {
                *count -= 1;
            }
        }
    }
}

impl EnhancedFederationManager {
    /// Create a new enhanced federation manager
    pub async fn new(
        config: EnhancedFederationConfig,
        local_schema: Arc<Schema>,
    ) -> Result<Self> {
        let service_discovery = Arc::new(ServiceDiscovery::new(config.service_discovery.clone()));
        let schema_stitcher = Arc::new(SchemaStitcher::new(local_schema));
        let federation_config = FederationConfig::default();
        let query_planner = Arc::new(QueryPlanner::new(schema_stitcher.clone(), federation_config));
        let load_balancer = Arc::new(Mutex::new(LoadBalancer::new(config.load_balancing.clone())));

        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        let manager = Self {
            config,
            service_discovery,
            schema_stitcher,
            query_planner,
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            load_balancer,
            merged_schema: Arc::new(RwLock::new(None)),
            http_client,
        };

        // Set up event handling
        manager.setup_event_handling().await?;

        Ok(manager)
    }

    /// Start the federation manager
    pub async fn start(&self) -> Result<()> {
        info!("Starting enhanced federation manager");

        // Start service discovery
        self.service_discovery.start().await?;

        // Build initial merged schema
        self.rebuild_schema().await?;

        Ok(())
    }

    /// Execute a federated GraphQL query
    pub async fn execute_query(
        &self,
        document: &Document,
        variables: HashMap<String, Value>,
        context: FederationContext,
    ) -> Result<FederationResult> {
        let start_time = Instant::now();
        
        debug!("Executing federated query: {}", context.request_id);

        // Get current schema
        let schema = {
            let schema_guard = self.merged_schema.read().await;
            schema_guard.clone().ok_or_else(|| anyhow!("No schema available"))?
        };

        // Plan the query
        let query_plan = self.query_planner
            .plan_query(document, &schema)
            .await
            .context("Failed to plan federated query")?;

        // Execute the plan
        let result = self.execute_query_plan(&query_plan, &context).await?;

        let execution_time = start_time.elapsed();
        debug!("Query executed in {:?}: {}", execution_time, context.request_id);

        Ok(result)
    }

    /// Get the current merged schema
    pub async fn get_schema(&self) -> Option<Schema> {
        let schema_guard = self.merged_schema.read().await;
        schema_guard.clone()
    }

    /// Get all discovered services
    pub async fn get_services(&self) -> Vec<ServiceInfo> {
        self.service_discovery.get_services().await
    }

    /// Get healthy services only
    pub async fn get_healthy_services(&self) -> Vec<ServiceInfo> {
        self.service_discovery.get_healthy_services().await
    }

    /// Execute a query plan
    async fn execute_query_plan(
        &self,
        plan: &QueryPlan,
        context: &FederationContext,
    ) -> Result<FederationResult> {
        let mut results = HashMap::new();
        let mut errors = Vec::new();

        // Execute each step of the plan
        for step in &plan.steps {
            match self.execute_query_step(step, context, &results).await {
                Ok(step_result) => {
                    if let Some(data) = step_result.data {
                        results.insert(step.endpoint_id.clone(), data);
                    }
                    errors.extend(step_result.errors);
                }
                Err(e) => {
                    errors.push(FederationError {
                        message: format!("Failed to execute step for service {}: {}", step.endpoint_id, e),
                        path: None,
                        locations: None,
                        extensions: HashMap::new(),
                    });
                }
            }
        }

        // Merge results
        let merged_data = self.merge_results(&results, plan).await?;

        Ok(FederationResult {
            data: Some(merged_data),
            errors,
            extensions: HashMap::new(),
        })
    }

    /// Execute a single query step
    async fn execute_query_step(
        &self,
        step: &crate::federation::query_planner::QueryStep,
        context: &FederationContext,
        _previous_results: &HashMap<String, serde_json::Value>,
    ) -> Result<FederationResult> {
        // Get service info
        let service = self.service_discovery
            .get_service(&step.endpoint_id)
            .await
            .ok_or_else(|| anyhow!("Service not found: {}", step.endpoint_id))?;

        // Check circuit breaker
        {
            let mut breakers = self.circuit_breakers.write().await;
            let breaker = breakers
                .entry(service.id.clone())
                .or_insert_with(|| ServiceCircuitBreaker::new(self.config.circuit_breaker.clone()));

            if !breaker.can_execute() {
                return Err(anyhow!("Circuit breaker open for service: {}", service.id));
            }
        }

        // Execute query with retry
        let result = self.execute_with_retry(&service, &step.query_fragment, &context.variables).await;

        // Update circuit breaker
        {
            let mut breakers = self.circuit_breakers.write().await;
            if let Some(breaker) = breakers.get_mut(&service.id) {
                match result {
                    Ok(_) => breaker.record_success(),
                    Err(_) => breaker.record_failure(),
                }
            }
        }

        // Record completion in load balancer
        self.load_balancer.lock().await.record_completion(&service.id);

        result
    }

    /// Execute query with retry logic
    async fn execute_with_retry(
        &self,
        service: &ServiceInfo,
        query: &str,
        variables: &HashMap<String, Value>,
    ) -> Result<FederationResult> {
        let mut last_error = None;

        for attempt in 0..=self.config.retry_policy.max_retries {
            if attempt > 0 {
                let delay = self.calculate_retry_delay(attempt);
                tokio::time::sleep(delay).await;
            }

            match self.execute_single_request(service, query, variables).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);
                    warn!(
                        "Query attempt {} failed for service {}: {}",
                        attempt + 1,
                        service.id,
                        last_error.as_ref().unwrap()
                    );
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("All retry attempts failed")))
    }

    /// Calculate retry delay with exponential backoff
    fn calculate_retry_delay(&self, attempt: usize) -> Duration {
        let base_delay = self.config.retry_policy.base_delay;
        let multiplier = self.config.retry_policy.backoff_multiplier;
        let max_delay = self.config.retry_policy.max_delay;

        let delay = base_delay.as_millis() as f64 * multiplier.powi(attempt as i32);
        let delay = Duration::from_millis(delay.min(max_delay.as_millis() as f64) as u64);

        if self.config.retry_policy.jitter {
            let jitter = fastrand::f64() * 0.1; // ±10% jitter
            let jitter_factor = 1.0 + (jitter - 0.05);
            Duration::from_millis((delay.as_millis() as f64 * jitter_factor) as u64)
        } else {
            delay
        }
    }

    /// Execute a single HTTP request to a service
    async fn execute_single_request(
        &self,
        service: &ServiceInfo,
        query: &str,
        variables: &HashMap<String, Value>,
    ) -> Result<FederationResult> {
        let request_body = serde_json::json!({
            "query": query,
            "variables": variables
        });

        let response = self.http_client
            .post(&service.url)
            .json(&request_body)
            .send()
            .await
            .context("Failed to send request")?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "HTTP request failed with status: {}",
                response.status()
            ));
        }

        let response_json: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse response JSON")?;

        // Parse GraphQL response
        let data = response_json.get("data").cloned();
        let errors = response_json
            .get("errors")
            .and_then(|e| e.as_array())
            .map(|errors| {
                errors
                    .iter()
                    .filter_map(|error| {
                        Some(FederationError {
                            message: error.get("message")?.as_str()?.to_string(),
                            path: error.get("path").and_then(|p| {
                                p.as_array().map(|arr| {
                                    arr.iter()
                                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                        .collect()
                                })
                            }),
                            locations: None, // Could be parsed from error object
                            extensions: HashMap::new(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(FederationResult {
            data,
            errors,
            extensions: HashMap::new(),
        })
    }

    /// Merge results from multiple services
    async fn merge_results(
        &self,
        results: &HashMap<String, serde_json::Value>,
        _plan: &QueryPlan,
    ) -> Result<serde_json::Value> {
        // Simple merge for now - in a real implementation this would be more sophisticated
        if results.len() == 1 {
            Ok(results.values().next().unwrap().clone())
        } else {
            // Merge multiple results
            let mut merged = serde_json::Map::new();
            for (_service_id, result) in results {
                if let Some(obj) = result.as_object() {
                    for (key, value) in obj {
                        merged.insert(key.clone(), value.clone());
                    }
                }
            }
            Ok(serde_json::Value::Object(merged))
        }
    }

    /// Rebuild the merged schema from all discovered services
    async fn rebuild_schema(&self) -> Result<()> {
        info!("Rebuilding federated schema");

        let services = self.service_discovery.get_healthy_services().await;
        let endpoints: Vec<RemoteEndpoint> = services
            .into_iter()
            .map(|service| RemoteEndpoint {
                id: service.id,
                url: service.url,
                namespace: Some(service.name),
                auth_header: None,
                timeout_secs: 30,
                max_retries: 3,
                retry_strategy: RetryStrategy::ExponentialBackoff {
                    initial_delay_ms: 100,
                    max_delay_ms: 5000,
                    multiplier: 2.0,
                },
                health_check_url: None,
                priority: 0,
                schema_version: service.federation_version,
                min_compatible_version: None,
            })
            .collect();

        let merged_schema = self.schema_stitcher.merge_schemas(&endpoints).await?;

        {
            let mut schema_guard = self.merged_schema.write().await;
            *schema_guard = Some(merged_schema);
        }

        info!("Schema rebuilt successfully");
        Ok(())
    }

    /// Set up event handling for service discovery
    async fn setup_event_handling(&self) -> Result<()> {
        let schema_rebuild_handler = SchemaRebuildHandler {
            manager: self as *const Self,
        };

        // Note: In a real implementation, we'd need to properly handle the lifetime
        // This is a simplified version for demonstration
        info!("Service discovery event handling configured");
        Ok(())
    }
}

/// Handler for schema rebuild events
struct SchemaRebuildHandler {
    manager: *const EnhancedFederationManager,
}

// Note: This is not safe in practice - just for demonstration
unsafe impl Send for SchemaRebuildHandler {}
unsafe impl Sync for SchemaRebuildHandler {}

#[async_trait]
impl ServiceDiscoveryEventHandler for SchemaRebuildHandler {
    async fn handle_event(&self, event: DiscoveryEvent) -> Result<()> {
        match event {
            DiscoveryEvent::ServiceRegistered(_) |
            DiscoveryEvent::ServiceDeregistered(_) |
            DiscoveryEvent::HealthChanged { .. } => {
                // In a real implementation, we'd safely access the manager
                info!("Service discovery event received, would trigger schema rebuild");
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Schema;

    #[tokio::test]
    async fn test_load_balancer_round_robin() {
        let config = LoadBalancingConfig {
            algorithm: LoadBalancingAlgorithm::RoundRobin,
            ..Default::default()
        };
        let mut balancer = LoadBalancer::new(config);

        let services = vec![
            ServiceInfo {
                id: "service-1".to_string(),
                name: "Service 1".to_string(),
                url: "http://localhost:4001".to_string(),
                version: None,
                capabilities: Default::default(),
                health_status: HealthStatus::Healthy,
                metadata: HashMap::new(),
                last_seen: chrono::Utc::now(),
                response_time: None,
                load_factor: 0.5,
                federation_version: None,
            },
            ServiceInfo {
                id: "service-2".to_string(),
                name: "Service 2".to_string(),
                url: "http://localhost:4002".to_string(),
                version: None,
                capabilities: Default::default(),
                health_status: HealthStatus::Healthy,
                metadata: HashMap::new(),
                last_seen: chrono::Utc::now(),
                response_time: None,
                load_factor: 0.5,
                federation_version: None,
            },
        ];

        let selected1 = balancer.select_service(&services, None);
        let selected2 = balancer.select_service(&services, None);

        assert_eq!(selected1, Some("service-1".to_string()));
        assert_eq!(selected2, Some("service-2".to_string()));
    }

    #[test]
    fn test_circuit_breaker() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 1,
            timeout: Duration::from_secs(1),
            retry_after: Duration::from_secs(1),
        };
        let mut breaker = ServiceCircuitBreaker::new(config);

        assert!(breaker.can_execute());
        assert_eq!(breaker.state(), CircuitBreakerState::Closed);

        // Record failures to open circuit
        breaker.record_failure();
        assert!(breaker.can_execute());
        breaker.record_failure();
        assert!(!breaker.can_execute());
        assert_eq!(breaker.state(), CircuitBreakerState::Open);
    }
}