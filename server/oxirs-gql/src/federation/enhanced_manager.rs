//! Enhanced Federation Manager with Dynamic Service Discovery
//!
//! This module provides an advanced federation manager that combines service discovery,
//! intelligent query routing, load balancing, and fault tolerance.

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

use super::config::{FederationConfig, RemoteEndpoint, RetryStrategy};
use super::query_planner::{QueryPlan, QueryPlanner, QueryStep};
use super::real_time_sync::{RealTimeSchemaSynchronizer, SyncConfig};
use super::schema_stitcher::SchemaStitcher;
use super::service_discovery::{
    DiscoveryEvent, HealthStatus, ServiceDiscovery, ServiceDiscoveryConfig,
    ServiceDiscoveryEventHandler, ServiceInfo,
};
use crate::ast::{Document, OperationType, Value};
use crate::types::Schema;

/// Enhanced federation manager configuration
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct EnhancedFederationConfig {
    pub service_discovery: ServiceDiscoveryConfig,
    pub load_balancing: LoadBalancingConfig,
    pub circuit_breaker: CircuitBreakerConfig,
    pub retry_policy: RetryPolicyConfig,
    pub query_routing: QueryRoutingConfig,
    pub caching: FederationCacheConfig,
    pub real_time_sync: SyncConfig,
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

/// Query performance analytics tracker
#[derive(Debug, Clone)]
pub struct QueryAnalytics {
    pub query_patterns: HashMap<String, QueryPatternStats>,
    pub service_performance: HashMap<String, ServicePerformanceStats>,
    pub recent_queries: Vec<QueryExecutionStats>,
    pub max_history: usize,
}

#[derive(Debug, Clone)]
pub struct QueryPatternStats {
    pub pattern_hash: String,
    pub execution_count: u64,
    pub average_duration: Duration,
    pub success_rate: f64,
    pub preferred_services: Vec<String>,
    pub complexity_score: f64,
}

#[derive(Debug, Clone)]
pub struct ServicePerformanceStats {
    pub service_id: String,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub average_response_time: Duration,
    pub error_rate: f64,
    pub query_type_performance: HashMap<String, Duration>,
    pub last_updated: Instant,
}

#[derive(Debug, Clone)]
pub struct QueryExecutionStats {
    pub query_hash: String,
    pub service_id: String,
    pub duration: Duration,
    pub success: bool,
    pub timestamp: Instant,
    pub complexity: f64,
    pub cache_hit: bool,
}

impl Default for QueryAnalytics {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryAnalytics {
    pub fn new() -> Self {
        Self {
            query_patterns: HashMap::new(),
            service_performance: HashMap::new(),
            recent_queries: Vec::new(),
            max_history: 10000,
        }
    }

    /// Record a query execution
    pub fn record_execution(&mut self, stats: QueryExecutionStats) {
        // Update query pattern statistics
        let pattern_stats = self
            .query_patterns
            .entry(stats.query_hash.clone())
            .or_insert_with(|| QueryPatternStats {
                pattern_hash: stats.query_hash.clone(),
                execution_count: 0,
                average_duration: Duration::from_millis(0),
                success_rate: 1.0,
                preferred_services: Vec::new(),
                complexity_score: stats.complexity,
            });

        pattern_stats.execution_count += 1;
        pattern_stats.average_duration = Duration::from_nanos(
            ((pattern_stats.average_duration.as_nanos() as u64
                * (pattern_stats.execution_count - 1))
                + stats.duration.as_nanos() as u64)
                / pattern_stats.execution_count,
        );

        if stats.success {
            pattern_stats.success_rate =
                (pattern_stats.success_rate * (pattern_stats.execution_count - 1) as f64 + 1.0)
                    / pattern_stats.execution_count as f64;
        } else {
            pattern_stats.success_rate = (pattern_stats.success_rate
                * (pattern_stats.execution_count - 1) as f64)
                / pattern_stats.execution_count as f64;
        }

        // Update service performance statistics
        let service_stats = self
            .service_performance
            .entry(stats.service_id.clone())
            .or_insert_with(|| ServicePerformanceStats {
                service_id: stats.service_id.clone(),
                total_requests: 0,
                successful_requests: 0,
                average_response_time: Duration::from_millis(0),
                error_rate: 0.0,
                query_type_performance: HashMap::new(),
                last_updated: Instant::now(),
            });

        service_stats.total_requests += 1;
        if stats.success {
            service_stats.successful_requests += 1;
        }

        service_stats.average_response_time = Duration::from_nanos(
            ((service_stats.average_response_time.as_nanos() as u64
                * (service_stats.total_requests - 1))
                + stats.duration.as_nanos() as u64)
                / service_stats.total_requests,
        );

        service_stats.error_rate =
            1.0 - (service_stats.successful_requests as f64 / service_stats.total_requests as f64);
        service_stats.last_updated = Instant::now();

        // Add to recent queries history
        self.recent_queries.push(stats);
        if self.recent_queries.len() > self.max_history {
            self.recent_queries
                .drain(0..self.recent_queries.len() - self.max_history);
        }
    }

    /// Get recommended service for a query pattern
    pub fn get_recommended_service(&self, query_hash: &str) -> Option<String> {
        if let Some(_pattern_stats) = self.query_patterns.get(query_hash) {
            // Find the service with best performance for this pattern
            let mut best_service = None;
            let mut best_score = f64::NEG_INFINITY;

            for (service_id, service_stats) in &self.service_performance {
                // Calculate a composite score based on success rate, response time, and error rate
                let success_weight = 0.4;
                let response_time_weight = 0.4;
                let error_rate_weight = 0.2;

                let success_score = service_stats.successful_requests as f64
                    / service_stats.total_requests.max(1) as f64;
                let response_time_score =
                    1.0 / (service_stats.average_response_time.as_millis().max(1) as f64);
                let error_score = 1.0 - service_stats.error_rate;

                let composite_score = (success_score * success_weight)
                    + (response_time_score * response_time_weight)
                    + (error_score * error_rate_weight);

                if composite_score > best_score {
                    best_score = composite_score;
                    best_service = Some(service_id.clone());
                }
            }

            best_service
        } else {
            None
        }
    }

    /// Get performance insights
    pub fn get_performance_insights(&self) -> HashMap<String, serde_json::Value> {
        let mut insights = HashMap::new();

        insights.insert(
            "total_query_patterns".to_string(),
            serde_json::Value::Number(self.query_patterns.len().into()),
        );

        insights.insert(
            "total_services".to_string(),
            serde_json::Value::Number(self.service_performance.len().into()),
        );

        insights.insert(
            "recent_queries_count".to_string(),
            serde_json::Value::Number(self.recent_queries.len().into()),
        );

        // Calculate overall system performance
        let total_requests: u64 = self
            .service_performance
            .values()
            .map(|s| s.total_requests)
            .sum();

        let total_successful: u64 = self
            .service_performance
            .values()
            .map(|s| s.successful_requests)
            .sum();

        if total_requests > 0 {
            insights.insert(
                "overall_success_rate".to_string(),
                serde_json::Value::Number(
                    serde_json::Number::from_f64(total_successful as f64 / total_requests as f64)
                        .unwrap_or_else(|| serde_json::Number::from(0)),
                ),
            );
        }

        insights
    }
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
    schema_synchronizer: Arc<RealTimeSchemaSynchronizer>,
    query_analytics: Arc<Mutex<QueryAnalytics>>,
    http_client: reqwest::Client,
}

/// Load balancer implementation
#[derive(Debug)]
pub struct LoadBalancer {
    algorithm: LoadBalancingAlgorithm,
    round_robin_counter: usize,
    request_counts: HashMap<String, usize>,
    config: LoadBalancingConfig,
    hash_ring: BTreeMap<u64, String>,
    virtual_nodes_per_service: usize,
}

impl LoadBalancer {
    pub fn new(config: LoadBalancingConfig) -> Self {
        Self {
            algorithm: config.algorithm.clone(),
            round_robin_counter: 0,
            request_counts: HashMap::new(),
            config,
            hash_ring: BTreeMap::new(),
            virtual_nodes_per_service: 150, // Standard for consistent hashing
        }
    }

    /// Update the hash ring with available services
    pub fn update_hash_ring(&mut self, services: &[ServiceInfo]) {
        self.hash_ring.clear();

        for service in services {
            // Create virtual nodes for better distribution
            for i in 0..self.virtual_nodes_per_service {
                let virtual_key = format!("{}:{}", service.id, i);
                let hash = self.hash_string(&virtual_key);
                self.hash_ring.insert(hash, service.id.clone());
            }
        }
    }

    /// Hash a string to u64
    fn hash_string(&self, s: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }

    /// Select the best service for a request
    pub fn select_service(
        &mut self,
        services: &[ServiceInfo],
        _query_hash: Option<u64>,
    ) -> Option<String> {
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
            LoadBalancingAlgorithm::Adaptive => self.select_adaptive(&healthy_services),
            LoadBalancingAlgorithm::ConsistentHashing => {
                // Update hash ring with current healthy services
                let healthy_service_infos: Vec<_> =
                    healthy_services.iter().map(|&s| s.clone()).collect();
                self.update_hash_ring(&healthy_service_infos);
                // Use the query_hash for consistent routing
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

            let response_time_score = service
                .response_time
                .map(|rt| rt.as_millis() as f64)
                .unwrap_or(1000.0);

            let load_score = service.load_factor;
            let request_count = self.request_counts.get(&service.id).copied().unwrap_or(0) as f64;

            let total_score = (health_score * self.config.health_check_weight)
                + (response_time_score * self.config.response_time_weight)
                + (load_score * self.config.load_weight)
                + (request_count * 0.1);

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
            .min_by_key(|service| self.request_counts.get(&service.id).copied().unwrap_or(0))
            .map(|service| service.id.clone())
    }

    fn select_least_response_time(&self, services: &[&ServiceInfo]) -> Option<String> {
        services
            .iter()
            .min_by_key(|service| service.response_time.unwrap_or(Duration::from_secs(10)))
            .map(|service| service.id.clone())
    }

    fn select_adaptive(&mut self, services: &[&ServiceInfo]) -> Option<String> {
        // Adaptive algorithm combines multiple factors
        self.select_weighted_round_robin(services)
    }

    fn select_consistent_hash(&self, _services: &[&ServiceInfo], hash: u64) -> Option<String> {
        if self.hash_ring.is_empty() {
            return None;
        }

        // Find the first node in the ring that is >= hash
        if let Some((_, service_id)) = self.hash_ring.range(hash..).next() {
            Some(service_id.clone())
        } else {
            // Wrap around to the first node in the ring
            self.hash_ring
                .first_key_value()
                .map(|(_, service_id)| service_id.clone())
        }
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
    pub async fn new(config: EnhancedFederationConfig, local_schema: Arc<Schema>) -> Result<Self> {
        let service_discovery = Arc::new(ServiceDiscovery::new(config.service_discovery.clone()));
        let schema_stitcher = Arc::new(SchemaStitcher::new(local_schema));
        let federation_config = FederationConfig::default();
        let query_planner = Arc::new(QueryPlanner::new(
            schema_stitcher.clone(),
            federation_config,
        ));
        let load_balancer = Arc::new(Mutex::new(LoadBalancer::new(config.load_balancing.clone())));

        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        let schema_synchronizer = Arc::new(RealTimeSchemaSynchronizer::new(
            config.real_time_sync.clone(),
            service_discovery.clone(),
            schema_stitcher.clone(),
        ));

        let manager = Self {
            config,
            service_discovery,
            schema_stitcher,
            query_planner,
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            load_balancer,
            merged_schema: Arc::new(RwLock::new(None)),
            schema_synchronizer,
            query_analytics: Arc::new(Mutex::new(QueryAnalytics::new())),
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

        // Start real-time schema synchronization
        self.schema_synchronizer.start().await?;

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

        // Generate query hash for analytics
        let query_hash = self.generate_query_hash(document, &variables);

        // Check analytics for recommended service
        let recommended_service = {
            let analytics = self.query_analytics.lock().await;
            analytics.get_recommended_service(&query_hash)
        };

        // Get current schema
        let schema = {
            let schema_guard = self.merged_schema.read().await;
            schema_guard
                .clone()
                .ok_or_else(|| anyhow!("No schema available"))?
        };

        // Plan the query
        let query_plan = self
            .query_planner
            .plan_query(document, &schema)
            .await
            .context("Failed to plan federated query")?;

        // Execute the plan with analytics-aware service selection
        let result = self
            .execute_query_plan_with_analytics(
                &query_plan,
                &context,
                &query_hash,
                recommended_service,
            )
            .await;

        let execution_time = start_time.elapsed();
        let success = result.is_ok();

        // Record execution statistics
        if let Ok(ref _plan_result) = result {
            // Record analytics for each service used
            for step in &query_plan.steps {
                let service_id = &step.endpoint_id;
                let stats = QueryExecutionStats {
                    query_hash: query_hash.clone(),
                    service_id: service_id.clone(),
                    duration: execution_time,
                    success,
                    timestamp: start_time,
                    complexity: self.calculate_query_complexity(document),
                    cache_hit: false, // This would be determined by cache layer
                };

                let mut analytics = self.query_analytics.lock().await;
                analytics.record_execution(stats);
            }
        }

        debug!(
            "Query executed in {:?}: {} (success: {})",
            execution_time, context.request_id, success
        );

        result
    }

    /// Generate a hash for the query for analytics tracking
    fn generate_query_hash(
        &self,
        document: &Document,
        variables: &HashMap<String, Value>,
    ) -> String {
        let mut hasher = DefaultHasher::new();

        // Create a simplified representation of the query for hashing
        let query_str = format!("{document:?}");
        query_str.hash(&mut hasher);

        // Include variables in the hash if they significantly affect query characteristics
        let variables_str = format!("{variables:?}");
        variables_str.hash(&mut hasher);

        format!("{:x}", hasher.finish())
    }

    /// Calculate query complexity score
    fn calculate_query_complexity(&self, document: &Document) -> f64 {
        let mut complexity = 0.0;

        // Basic complexity calculation based on operation type and depth
        for definition in &document.definitions {
            match definition {
                crate::ast::Definition::Operation(op) => {
                    complexity += match op.operation_type {
                        OperationType::Query => 1.0,
                        OperationType::Mutation => 2.0,
                        OperationType::Subscription => 1.5,
                    };

                    // Add complexity based on selection depth
                    complexity += self.calculate_selection_complexity(&op.selection_set, 0) as f64;
                }
                crate::ast::Definition::Fragment(_) => {
                    complexity += 0.5; // Fragments add some complexity
                }
                crate::ast::Definition::Schema(_) => {
                    complexity += 0.1; // Schema definitions add minimal complexity
                }
                crate::ast::Definition::Type(_) => {
                    complexity += 0.1; // Type definitions add minimal complexity
                }
                crate::ast::Definition::Directive(_) => {
                    complexity += 0.1; // Directive definitions add minimal complexity
                }
                crate::ast::Definition::SchemaExtension(_) => {
                    complexity += 0.1; // Schema extensions add minimal complexity
                }
                crate::ast::Definition::TypeExtension(_) => {
                    complexity += 0.1; // Type extensions add minimal complexity
                }
            }
        }

        complexity
    }

    /// Calculate complexity of a selection set recursively
    fn calculate_selection_complexity(
        &self,
        selection_set: &crate::ast::SelectionSet,
        depth: usize,
    ) -> usize {
        let mut complexity = depth * 2; // Depth multiplier

        for selection in &selection_set.selections {
            match selection {
                crate::ast::Selection::Field(field) => {
                    complexity += 1;
                    if let Some(ref selection_set) = field.selection_set {
                        complexity += self.calculate_selection_complexity(selection_set, depth + 1);
                    }
                    // Add complexity for arguments
                    complexity += field.arguments.len();
                }
                crate::ast::Selection::InlineFragment(fragment) => {
                    complexity +=
                        self.calculate_selection_complexity(&fragment.selection_set, depth + 1);
                }
                crate::ast::Selection::FragmentSpread(_) => {
                    complexity += 2; // Fragment spreads add complexity
                }
            }
        }

        complexity
    }

    /// Execute query plan with analytics-aware service selection
    async fn execute_query_plan_with_analytics(
        &self,
        query_plan: &QueryPlan,
        context: &FederationContext,
        _query_hash: &str,
        recommended_service: Option<String>,
    ) -> Result<FederationResult> {
        // If we have a recommended service from analytics, try to use it
        if let Some(service_id) = recommended_service {
            debug!("Using analytics-recommended service: {}", service_id);
        }

        // For now, delegate to the original query plan execution
        // In a full implementation, this would integrate the recommended service
        // into the service selection logic
        self.execute_query_plan(query_plan, context).await
    }

    /// Get performance analytics insights
    pub async fn get_analytics_insights(&self) -> HashMap<String, serde_json::Value> {
        let analytics = self.query_analytics.lock().await;
        analytics.get_performance_insights()
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

    /// Get schema synchronization status
    pub async fn get_sync_status(&self) -> super::real_time_sync::SyncStatus {
        self.schema_synchronizer.get_sync_status().await
    }

    /// Get active schema conflicts
    pub async fn get_schema_conflicts(&self) -> Vec<super::real_time_sync::SchemaConflict> {
        self.schema_synchronizer.get_active_conflicts().await
    }

    /// Force a manual schema synchronization
    pub async fn force_schema_sync(&self) -> Result<()> {
        self.schema_synchronizer.perform_full_sync().await
    }

    /// Subscribe to schema changes
    pub async fn subscribe_to_schema_changes(
        &self,
    ) -> tokio::sync::mpsc::UnboundedReceiver<super::real_time_sync::SchemaChangeEvent> {
        self.schema_synchronizer.subscribe_to_changes().await
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
                        message: format!(
                            "Failed to execute step for service {}: {}",
                            step.endpoint_id, e
                        ),
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
        step: &QueryStep,
        context: &FederationContext,
        _previous_results: &HashMap<String, serde_json::Value>,
    ) -> Result<FederationResult> {
        // Get service info
        let service = self
            .service_discovery
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
        let result = self
            .execute_with_retry(&service, &step.query_fragment, &context.variables)
            .await;

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
        self.load_balancer
            .lock()
            .await
            .record_completion(&service.id);

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

        let response = self
            .http_client
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
            for result in results.values() {
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
        let _schema_rebuild_handler = SchemaRebuildHandler {
            manager: self as *const Self,
        };

        // Note: In a real implementation, we'd need to properly handle the lifetime
        // This is a simplified version for demonstration
        info!("Service discovery event handling configured");
        Ok(())
    }
}

/// Handler for schema rebuild events
#[allow(dead_code)]
struct SchemaRebuildHandler {
    #[allow(dead_code)]
    manager: *const EnhancedFederationManager,
}

// Note: This is not safe in practice - just for demonstration
unsafe impl Send for SchemaRebuildHandler {}
unsafe impl Sync for SchemaRebuildHandler {}

#[async_trait]
impl ServiceDiscoveryEventHandler for SchemaRebuildHandler {
    async fn handle_event(&self, event: DiscoveryEvent) -> Result<()> {
        match event {
            DiscoveryEvent::ServiceRegistered(_)
            | DiscoveryEvent::ServiceDeregistered(_)
            | DiscoveryEvent::HealthChanged { .. } => {
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
