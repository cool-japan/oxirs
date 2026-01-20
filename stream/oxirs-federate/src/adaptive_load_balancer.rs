//! Adaptive Load Balancer for Federated Services
//!
//! This module provides intelligent load balancing for federated SPARQL and GraphQL services,
//! with adaptive algorithms that learn from performance patterns and dynamically adjust
//! query distribution based on service health, capacity, and current load.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Type alias for performance history entries
type PerformanceHistory = Arc<RwLock<VecDeque<(String, Duration, bool, SystemTime)>>>;

/// Load balancing strategy
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Weighted round-robin based on service capacity
    WeightedRoundRobin,
    /// Least connections algorithm
    LeastConnections,
    /// Least response time algorithm
    LeastResponseTime,
    /// Adaptive algorithm based on performance metrics
    Adaptive,
    /// Consistent hashing for cache affinity
    ConsistentHashing,
    /// Load-aware distribution
    LoadAware,
}

/// Configuration for the adaptive load balancer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerConfig {
    /// Primary load balancing strategy
    pub strategy: LoadBalancingStrategy,
    /// Fallback strategy if primary fails
    pub fallback_strategy: LoadBalancingStrategy,
    /// Enable health-based filtering
    pub enable_health_filtering: bool,
    /// Minimum healthy services before degraded mode
    pub min_healthy_services: usize,
    /// Performance window for adaptive decisions
    pub performance_window: Duration,
    /// Weight adjustment sensitivity
    pub weight_adjustment_rate: f64,
    /// Circuit breaker threshold
    pub circuit_breaker_threshold: f64,
    /// Circuit breaker timeout
    pub circuit_breaker_timeout: Duration,
    /// Enable query affinity (consistent routing for similar queries)
    pub enable_query_affinity: bool,
    /// Affinity cache size
    pub affinity_cache_size: usize,
    /// Maximum concurrent connections per service
    pub max_connections_per_service: usize,
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            strategy: LoadBalancingStrategy::Adaptive,
            fallback_strategy: LoadBalancingStrategy::LeastConnections,
            enable_health_filtering: true,
            min_healthy_services: 1,
            performance_window: Duration::from_secs(300), // 5 minutes
            weight_adjustment_rate: 0.1,
            circuit_breaker_threshold: 0.5, // 50% error rate
            circuit_breaker_timeout: Duration::from_secs(60),
            enable_query_affinity: true,
            affinity_cache_size: 1000,
            max_connections_per_service: 100,
        }
    }
}

/// Service health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceHealth {
    Healthy,
    Degraded,
    Unhealthy,
    CircuitOpen,
}

/// Real-time service metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMetrics {
    /// Service identifier
    pub service_id: String,
    /// Current health status
    pub health: ServiceHealth,
    /// Current active connections
    pub active_connections: usize,
    /// Average response time over window
    pub avg_response_time: Duration,
    /// Request success rate
    pub success_rate: f64,
    /// Current load score (0.0 = no load, 1.0 = fully loaded)
    pub load_score: f64,
    /// Service weight for load balancing
    pub weight: f64,
    /// Last health check timestamp
    pub last_health_check: SystemTime,
    /// Circuit breaker state
    pub circuit_breaker_open_until: Option<SystemTime>,
    /// Recent error count
    pub recent_errors: usize,
    /// Recent request count
    pub recent_requests: usize,
    /// Service capacity estimate
    pub estimated_capacity: usize,
    /// Geographic location (for latency optimization)
    pub location: Option<String>,
}

impl ServiceMetrics {
    /// Create new service metrics
    pub fn new(service_id: String) -> Self {
        Self {
            service_id,
            health: ServiceHealth::Healthy,
            active_connections: 0,
            avg_response_time: Duration::from_millis(100),
            success_rate: 1.0,
            load_score: 0.0,
            weight: 1.0,
            last_health_check: SystemTime::now(),
            circuit_breaker_open_until: None,
            recent_errors: 0,
            recent_requests: 0,
            estimated_capacity: 100,
            location: None,
        }
    }

    /// Check if service is available for new requests
    pub fn is_available(&self) -> bool {
        match self.health {
            ServiceHealth::Healthy | ServiceHealth::Degraded => {
                if let Some(open_until) = self.circuit_breaker_open_until {
                    SystemTime::now() > open_until
                } else {
                    self.active_connections < self.estimated_capacity
                }
            }
            _ => false,
        }
    }

    /// Calculate service selection score
    pub fn calculate_selection_score(&self) -> f64 {
        if !self.is_available() {
            return 0.0;
        }

        let health_factor = match self.health {
            ServiceHealth::Healthy => 1.0,
            ServiceHealth::Degraded => 0.7,
            _ => 0.0,
        };

        let load_factor = 1.0 - self.load_score;
        let response_time_factor = 1.0 / (self.avg_response_time.as_millis() as f64 + 1.0);
        let success_factor = self.success_rate;
        let capacity_factor = (self.estimated_capacity - self.active_connections) as f64
            / self.estimated_capacity as f64;

        health_factor
            * load_factor
            * response_time_factor
            * success_factor
            * capacity_factor
            * self.weight
    }

    /// Update metrics from query execution
    pub fn update_from_execution(&mut self, response_time: Duration, success: bool) {
        self.recent_requests += 1;
        if !success {
            self.recent_errors += 1;
        }

        // Update average response time (exponential moving average)
        let alpha = 0.2; // Smoothing factor
        let current_ms = self.avg_response_time.as_millis() as f64;
        let new_ms = response_time.as_millis() as f64;
        let updated_ms = alpha * new_ms + (1.0 - alpha) * current_ms;
        self.avg_response_time = Duration::from_millis(updated_ms as u64);

        // Update success rate
        self.success_rate =
            (self.recent_requests - self.recent_errors) as f64 / self.recent_requests as f64;

        // Update load score based on connections and response time
        let connection_load = self.active_connections as f64 / self.estimated_capacity as f64;
        let response_load = (response_time.as_millis() as f64 / 1000.0).min(1.0); // Normalize to 1s max
        self.load_score = (connection_load * 0.6 + response_load * 0.4).min(1.0);

        // Check for circuit breaker conditions
        if self.recent_requests >= 10 && self.success_rate < 0.5 {
            self.circuit_breaker_open_until = Some(SystemTime::now() + Duration::from_secs(60));
            self.health = ServiceHealth::CircuitOpen;
        }
    }
}

/// Query affinity cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffinityEntry {
    pub service_id: String,
    pub query_hash: u64,
    pub last_used: SystemTime,
    pub performance_score: f64,
}

/// Load balancer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerStats {
    pub total_requests: u64,
    pub requests_per_service: HashMap<String, u64>,
    pub avg_selection_time: Duration,
    pub health_check_failures: u64,
    pub circuit_breaker_activations: u64,
    pub affinity_cache_hits: u64,
    pub load_balancing_decisions: HashMap<LoadBalancingStrategy, u64>,
}

impl Default for LoadBalancerStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            requests_per_service: HashMap::new(),
            avg_selection_time: Duration::from_micros(0),
            health_check_failures: 0,
            circuit_breaker_activations: 0,
            affinity_cache_hits: 0,
            load_balancing_decisions: HashMap::new(),
        }
    }
}

/// Adaptive load balancer for federated services
pub struct AdaptiveLoadBalancer {
    config: LoadBalancerConfig,
    service_metrics: Arc<RwLock<HashMap<String, ServiceMetrics>>>,
    affinity_cache: Arc<RwLock<HashMap<u64, AffinityEntry>>>,
    round_robin_state: Arc<RwLock<usize>>,
    statistics: Arc<RwLock<LoadBalancerStats>>,
    performance_history: PerformanceHistory,
}

impl AdaptiveLoadBalancer {
    /// Create a new adaptive load balancer
    pub fn new(config: LoadBalancerConfig) -> Self {
        Self {
            config,
            service_metrics: Arc::new(RwLock::new(HashMap::new())),
            affinity_cache: Arc::new(RwLock::new(HashMap::new())),
            round_robin_state: Arc::new(RwLock::new(0)),
            statistics: Arc::new(RwLock::new(LoadBalancerStats::default())),
            performance_history: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    /// Register a new service with the load balancer
    pub async fn register_service(&self, service_id: String, initial_capacity: Option<usize>) {
        let mut metrics = ServiceMetrics::new(service_id.clone());
        if let Some(capacity) = initial_capacity {
            metrics.estimated_capacity = capacity;
        }

        self.service_metrics
            .write()
            .await
            .insert(service_id.clone(), metrics);
        info!("Registered service {} with load balancer", service_id);
    }

    /// Remove a service from the load balancer
    pub async fn deregister_service(&self, service_id: &str) {
        self.service_metrics.write().await.remove(service_id);
        info!("Deregistered service {} from load balancer", service_id);
    }

    /// Select the best service for a query
    pub async fn select_service(
        &self,
        query_hash: Option<u64>,
        _query_type: &str,
    ) -> Result<String> {
        let start_time = Instant::now();

        // Check query affinity cache first
        if let Some(hash) = query_hash {
            if self.config.enable_query_affinity {
                if let Some(service_id) = self.check_affinity_cache(hash).await {
                    self.record_affinity_hit().await;
                    return Ok(service_id);
                }
            }
        }

        let service_id = match self.config.strategy {
            LoadBalancingStrategy::RoundRobin => self.round_robin_selection().await,
            LoadBalancingStrategy::WeightedRoundRobin => {
                self.weighted_round_robin_selection().await
            }
            LoadBalancingStrategy::LeastConnections => self.least_connections_selection().await,
            LoadBalancingStrategy::LeastResponseTime => self.least_response_time_selection().await,
            LoadBalancingStrategy::Adaptive => self.adaptive_selection().await,
            LoadBalancingStrategy::ConsistentHashing => {
                if let Some(hash) = query_hash {
                    self.consistent_hash_selection(hash).await
                } else {
                    self.adaptive_selection().await // Fallback
                }
            }
            LoadBalancingStrategy::LoadAware => self.load_aware_selection().await,
        };

        let service_id = match service_id {
            Ok(id) => id,
            Err(_) => {
                warn!("Primary strategy failed, using fallback");
                self.fallback_selection().await?
            }
        };

        // Update affinity cache if enabled
        if let Some(hash) = query_hash {
            if self.config.enable_query_affinity {
                self.update_affinity_cache(hash, service_id.clone()).await;
            }
        }

        // Record selection statistics
        let selection_time = start_time.elapsed();
        self.record_selection(service_id.clone(), selection_time)
            .await;

        Ok(service_id)
    }

    /// Round-robin service selection
    async fn round_robin_selection(&self) -> Result<String> {
        let services = self.get_available_services().await;
        if services.is_empty() {
            return Err(anyhow!("No available services"));
        }

        let mut state = self.round_robin_state.write().await;
        let index = *state % services.len();
        *state = (*state + 1) % services.len();

        Ok(services[index].clone())
    }

    /// Weighted round-robin service selection
    async fn weighted_round_robin_selection(&self) -> Result<String> {
        let metrics = self.service_metrics.read().await;
        let available_services: Vec<_> = metrics.values().filter(|m| m.is_available()).collect();

        if available_services.is_empty() {
            return Err(anyhow!("No available services"));
        }

        // Calculate weighted selection
        let total_weight: f64 = available_services.iter().map(|m| m.weight).sum();
        let mut state = self.round_robin_state.write().await;
        let target = (*state as f64 * total_weight / 1000.0) % total_weight;
        *state = (*state + 1) % 1000;

        let mut cumulative_weight = 0.0;
        for service in &available_services {
            cumulative_weight += service.weight;
            if cumulative_weight >= target {
                return Ok(service.service_id.clone());
            }
        }

        // Fallback to first service
        Ok(available_services[0].service_id.clone())
    }

    /// Least connections service selection
    async fn least_connections_selection(&self) -> Result<String> {
        let metrics = self.service_metrics.read().await;
        let available_services: Vec<_> = metrics.values().filter(|m| m.is_available()).collect();

        if available_services.is_empty() {
            return Err(anyhow!("No available services"));
        }

        let best_service = available_services
            .iter()
            .min_by_key(|m| m.active_connections)
            .unwrap();

        Ok(best_service.service_id.clone())
    }

    /// Least response time service selection
    async fn least_response_time_selection(&self) -> Result<String> {
        let metrics = self.service_metrics.read().await;
        let available_services: Vec<_> = metrics.values().filter(|m| m.is_available()).collect();

        if available_services.is_empty() {
            return Err(anyhow!("No available services"));
        }

        let best_service = available_services
            .iter()
            .min_by_key(|m| m.avg_response_time)
            .unwrap();

        Ok(best_service.service_id.clone())
    }

    /// Adaptive service selection based on multiple factors
    async fn adaptive_selection(&self) -> Result<String> {
        let metrics = self.service_metrics.read().await;
        let available_services: Vec<_> = metrics.values().filter(|m| m.is_available()).collect();

        if available_services.is_empty() {
            return Err(anyhow!("No available services"));
        }

        let best_service = available_services
            .iter()
            .max_by(|a, b| {
                a.calculate_selection_score()
                    .partial_cmp(&b.calculate_selection_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        Ok(best_service.service_id.clone())
    }

    /// Consistent hash-based service selection
    async fn consistent_hash_selection(&self, query_hash: u64) -> Result<String> {
        let services = self.get_available_services().await;
        if services.is_empty() {
            return Err(anyhow!("No available services"));
        }

        // Simple consistent hashing implementation
        let index = (query_hash as usize) % services.len();
        Ok(services[index].clone())
    }

    /// Load-aware service selection
    async fn load_aware_selection(&self) -> Result<String> {
        let metrics = self.service_metrics.read().await;
        let available_services: Vec<_> = metrics.values().filter(|m| m.is_available()).collect();

        if available_services.is_empty() {
            return Err(anyhow!("No available services"));
        }

        // Select service with lowest load score
        let best_service = available_services
            .iter()
            .min_by(|a, b| {
                a.load_score
                    .partial_cmp(&b.load_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        Ok(best_service.service_id.clone())
    }

    /// Fallback service selection
    async fn fallback_selection(&self) -> Result<String> {
        match self.config.fallback_strategy {
            LoadBalancingStrategy::LeastConnections => self.least_connections_selection().await,
            LoadBalancingStrategy::RoundRobin => self.round_robin_selection().await,
            _ => {
                let services = self.get_available_services().await;
                if services.is_empty() {
                    Err(anyhow!("No available services"))
                } else {
                    Ok(services[0].clone())
                }
            }
        }
    }

    /// Get list of available service IDs
    async fn get_available_services(&self) -> Vec<String> {
        let metrics = self.service_metrics.read().await;
        metrics
            .values()
            .filter(|m| m.is_available())
            .map(|m| m.service_id.clone())
            .collect()
    }

    /// Check affinity cache for query
    async fn check_affinity_cache(&self, query_hash: u64) -> Option<String> {
        let cache = self.affinity_cache.read().await;
        if let Some(entry) = cache.get(&query_hash) {
            // Check if entry is still valid and service is available
            if let Ok(elapsed) = entry.last_used.elapsed() {
                if elapsed < Duration::from_secs(300) {
                    // 5 minute TTL
                    let metrics = self.service_metrics.read().await;
                    if let Some(service_metrics) = metrics.get(&entry.service_id) {
                        if service_metrics.is_available() {
                            return Some(entry.service_id.clone());
                        }
                    }
                }
            }
        }
        None
    }

    /// Update affinity cache
    async fn update_affinity_cache(&self, query_hash: u64, service_id: String) {
        let mut cache = self.affinity_cache.write().await;

        // Clean cache if it's full
        if cache.len() >= self.config.affinity_cache_size {
            // Remove oldest entries
            let cutoff = SystemTime::now() - Duration::from_secs(300);
            cache.retain(|_, entry| entry.last_used > cutoff);

            // If still full, remove random entries
            if cache.len() >= self.config.affinity_cache_size {
                let keys_to_remove: Vec<_> = cache.keys().take(cache.len() / 10).cloned().collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }
        }

        cache.insert(
            query_hash,
            AffinityEntry {
                service_id,
                query_hash,
                last_used: SystemTime::now(),
                performance_score: 1.0,
            },
        );
    }

    /// Record query execution result
    pub async fn record_execution(&self, service_id: &str, response_time: Duration, success: bool) {
        // Update service metrics
        if let Some(metrics) = self.service_metrics.write().await.get_mut(service_id) {
            metrics.update_from_execution(response_time, success);
        }

        // Add to performance history
        let mut history = self.performance_history.write().await;
        history.push_back((
            service_id.to_string(),
            response_time,
            success,
            SystemTime::now(),
        ));

        // Limit history size
        while history.len() > 10000 {
            history.pop_front();
        }
    }

    /// Mark service connection as started
    pub async fn start_connection(&self, service_id: &str) {
        if let Some(metrics) = self.service_metrics.write().await.get_mut(service_id) {
            metrics.active_connections += 1;
        }
    }

    /// Mark service connection as ended
    pub async fn end_connection(&self, service_id: &str) {
        if let Some(metrics) = self.service_metrics.write().await.get_mut(service_id) {
            if metrics.active_connections > 0 {
                metrics.active_connections -= 1;
            }
        }
    }

    /// Record affinity cache hit
    async fn record_affinity_hit(&self) {
        self.statistics.write().await.affinity_cache_hits += 1;
    }

    /// Record service selection
    async fn record_selection(&self, service_id: String, selection_time: Duration) {
        let mut stats = self.statistics.write().await;
        stats.total_requests += 1;
        *stats.requests_per_service.entry(service_id).or_insert(0) += 1;

        // Update average selection time
        let current_avg_ns = stats.avg_selection_time.as_nanos() as f64;
        let new_time_ns = selection_time.as_nanos() as f64;
        let updated_avg_ns = (current_avg_ns * (stats.total_requests - 1) as f64 + new_time_ns)
            / stats.total_requests as f64;
        stats.avg_selection_time = Duration::from_nanos(updated_avg_ns as u64);
    }

    /// Get current load balancer statistics
    pub async fn get_statistics(&self) -> LoadBalancerStats {
        self.statistics.read().await.clone()
    }

    /// Get service metrics
    pub async fn get_service_metrics(&self) -> HashMap<String, ServiceMetrics> {
        self.service_metrics.read().await.clone()
    }

    /// Perform health check on all services
    pub async fn health_check(&self) -> Result<()> {
        let mut metrics = self.service_metrics.write().await;
        let now = SystemTime::now();

        for (service_id, service_metrics) in metrics.iter_mut() {
            // Reset circuit breaker if timeout expired
            if let Some(open_until) = service_metrics.circuit_breaker_open_until {
                if now > open_until {
                    service_metrics.circuit_breaker_open_until = None;
                    service_metrics.health = ServiceHealth::Healthy;
                    info!("Circuit breaker reset for service: {}", service_id);
                }
            }

            // Update health based on recent performance
            if service_metrics.recent_requests >= 5 {
                if service_metrics.success_rate >= 0.9 {
                    service_metrics.health = ServiceHealth::Healthy;
                } else if service_metrics.success_rate >= 0.7 {
                    service_metrics.health = ServiceHealth::Degraded;
                } else {
                    service_metrics.health = ServiceHealth::Unhealthy;
                }
            }

            service_metrics.last_health_check = now;
        }

        Ok(())
    }

    /// Adjust service weights based on performance
    pub async fn adjust_weights(&self) -> Result<()> {
        let mut metrics = self.service_metrics.write().await;
        let history = self.performance_history.read().await;

        // Calculate performance baselines
        let mut service_performance: HashMap<String, (f64, usize)> = HashMap::new();

        let cutoff = SystemTime::now() - self.config.performance_window;
        for (service_id, response_time, success, timestamp) in history.iter() {
            if *timestamp > cutoff && *success {
                let (total_time, count) = service_performance
                    .entry(service_id.clone())
                    .or_insert((0.0, 0));
                *total_time += response_time.as_millis() as f64;
                *count += 1;
            }
        }

        // Calculate average response times
        let avg_response_times: HashMap<String, f64> = service_performance
            .into_iter()
            .map(|(service, (total, count))| (service, total / count as f64))
            .collect();

        if avg_response_times.is_empty() {
            return Ok(());
        }

        let overall_avg =
            avg_response_times.values().sum::<f64>() / avg_response_times.len() as f64;

        // Adjust weights based on relative performance
        for (_service_id, service_metrics) in metrics.iter_mut() {
            if let Some(&avg_time) = avg_response_times.get(&service_metrics.service_id) {
                let performance_ratio = overall_avg / avg_time;
                let target_weight = performance_ratio.clamp(0.1, 2.0); // Limit weight range

                // Gradually adjust weight
                let weight_diff = target_weight - service_metrics.weight;
                service_metrics.weight += weight_diff * self.config.weight_adjustment_rate;
                service_metrics.weight = service_metrics.weight.clamp(0.1, 2.0);
            }
        }

        debug!("Adjusted service weights based on performance");
        Ok(())
    }
}

impl Default for AdaptiveLoadBalancer {
    fn default() -> Self {
        Self::new(LoadBalancerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_registration() {
        let lb = AdaptiveLoadBalancer::default();

        lb.register_service("service-1".to_string(), Some(50)).await;
        lb.register_service("service-2".to_string(), Some(100))
            .await;

        let metrics = lb.get_service_metrics().await;
        assert_eq!(metrics.len(), 2);
        assert!(metrics.contains_key("service-1"));
        assert!(metrics.contains_key("service-2"));
    }

    #[tokio::test]
    async fn test_round_robin_selection() {
        let config = LoadBalancerConfig {
            strategy: LoadBalancingStrategy::RoundRobin,
            ..Default::default()
        };
        let lb = AdaptiveLoadBalancer::new(config);

        lb.register_service("service-1".to_string(), None).await;
        lb.register_service("service-2".to_string(), None).await;

        let first = lb.select_service(None, "test").await.unwrap();
        let second = lb.select_service(None, "test").await.unwrap();
        let third = lb.select_service(None, "test").await.unwrap();

        // Should cycle through services
        assert_ne!(first, second);
        assert_eq!(first, third);
    }

    #[tokio::test]
    async fn test_adaptive_selection() {
        let lb = AdaptiveLoadBalancer::default();

        lb.register_service("fast-service".to_string(), Some(100))
            .await;
        lb.register_service("slow-service".to_string(), Some(50))
            .await;

        // Simulate different performance
        lb.record_execution("fast-service", Duration::from_millis(50), true)
            .await;
        lb.record_execution("slow-service", Duration::from_millis(200), true)
            .await;

        let selected = lb.select_service(None, "test").await.unwrap();
        // Should prefer the faster service
        assert_eq!(selected, "fast-service");
    }

    #[tokio::test]
    async fn test_service_metrics_update() {
        let mut metrics = ServiceMetrics::new("test-service".to_string());

        assert_eq!(metrics.success_rate, 1.0);
        assert_eq!(metrics.recent_requests, 0);

        metrics.update_from_execution(Duration::from_millis(100), true);
        assert_eq!(metrics.success_rate, 1.0);
        assert_eq!(metrics.recent_requests, 1);

        metrics.update_from_execution(Duration::from_millis(200), false);
        assert_eq!(metrics.success_rate, 0.5);
        assert_eq!(metrics.recent_requests, 2);
    }
}
