// Advanced load balancing strategies for OxiRS Fuseki
//
// Provides multiple load balancing algorithms for distributing requests
// across federation endpoints, cluster nodes, or backend services.

use crate::error::{FusekiError, FusekiResult};
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use scirs2_core::random::{Random, StdRng};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

/// Load balancing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Round-robin: Distributes requests evenly across backends
    RoundRobin,
    /// Weighted round-robin: Distribution based on backend weights
    WeightedRoundRobin,
    /// Least connections: Route to backend with fewest active connections
    LeastConnections,
    /// Least response time: Route to fastest backend
    LeastResponseTime,
    /// Random: Randomly select a backend
    Random,
    /// Weighted random: Random selection based on weights
    WeightedRandom,
    /// IP hash: Consistent routing based on client IP
    IpHash,
    /// Consistent hashing: For cache distribution
    ConsistentHash,
    /// Power of two choices: Select best of 2 random backends
    PowerOfTwoChoices,
}

/// Backend server/endpoint
#[derive(Debug, Clone)]
pub struct Backend {
    pub id: String,
    pub url: String,
    pub weight: u32,
    pub max_connections: usize,
    pub health_check_url: Option<String>,
    pub enabled: bool,
}

/// Backend health status
#[derive(Debug, Clone)]
pub struct BackendHealth {
    pub backend_id: String,
    pub is_healthy: bool,
    pub last_check: Instant,
    pub consecutive_failures: u32,
    pub response_time_ms: u64,
}

/// Backend statistics
#[derive(Debug)]
struct BackendStats {
    active_connections: AtomicUsize,
    total_requests: AtomicU64,
    total_errors: AtomicU64,
    avg_response_time_ms: RwLock<f64>,
    last_request_time: RwLock<Option<Instant>>,
}

impl BackendStats {
    fn new() -> Self {
        Self {
            active_connections: AtomicUsize::new(0),
            total_requests: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            avg_response_time_ms: RwLock::new(0.0),
            last_request_time: RwLock::new(None),
        }
    }

    fn record_request(&self, response_time_ms: u64, is_error: bool) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        if is_error {
            self.total_errors.fetch_add(1, Ordering::Relaxed);
        }

        // Update average response time (exponential moving average)
        let mut avg = self.avg_response_time_ms.write();
        let alpha = 0.2; // Smoothing factor
        *avg = alpha * response_time_ms as f64 + (1.0 - alpha) * *avg;

        *self.last_request_time.write() = Some(Instant::now());
    }

    fn increment_connections(&self) {
        self.active_connections.fetch_add(1, Ordering::Relaxed);
    }

    fn decrement_connections(&self) {
        self.active_connections.fetch_sub(1, Ordering::Relaxed);
    }

    fn get_active_connections(&self) -> usize {
        self.active_connections.load(Ordering::Relaxed)
    }

    fn get_avg_response_time(&self) -> f64 {
        *self.avg_response_time_ms.read()
    }
}

/// Load balancer configuration
#[derive(Debug, Clone)]
pub struct LoadBalancerConfig {
    pub strategy: LoadBalancingStrategy,
    pub health_check_interval_secs: u64,
    pub health_check_timeout_secs: u64,
    pub max_failures_before_unhealthy: u32,
    pub enable_sticky_sessions: bool,
    pub session_cookie_name: String,
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            strategy: LoadBalancingStrategy::RoundRobin,
            health_check_interval_secs: 10,
            health_check_timeout_secs: 5,
            max_failures_before_unhealthy: 3,
            enable_sticky_sessions: false,
            session_cookie_name: "OXIRS_SESSION".to_string(),
        }
    }
}

/// Advanced load balancer
pub struct LoadBalancer {
    config: LoadBalancerConfig,
    backends: Arc<RwLock<Vec<Backend>>>,
    backend_stats: Arc<DashMap<String, Arc<BackendStats>>>,
    backend_health: Arc<DashMap<String, BackendHealth>>,
    round_robin_index: Arc<AtomicUsize>,
    random: Arc<Mutex<StdRng>>,
    session_affinity: Arc<DashMap<String, String>>, // session_id -> backend_id
}

impl LoadBalancer {
    /// Create a new load balancer
    pub fn new(config: LoadBalancerConfig) -> Self {
        // Generate seed from system time for randomness
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);

        Self {
            config,
            backends: Arc::new(RwLock::new(Vec::new())),
            backend_stats: Arc::new(DashMap::new()),
            backend_health: Arc::new(DashMap::new()),
            round_robin_index: Arc::new(AtomicUsize::new(0)),
            random: Arc::new(Mutex::new(Random::seed(seed))),
            session_affinity: Arc::new(DashMap::new()),
        }
    }

    /// Add a backend
    pub fn add_backend(&self, backend: Backend) -> FusekiResult<()> {
        let backend_id = backend.id.clone();

        // Add to backends list
        self.backends.write().push(backend.clone());

        // Initialize stats
        self.backend_stats
            .insert(backend_id.clone(), Arc::new(BackendStats::new()));

        // Initialize health status
        self.backend_health.insert(
            backend_id.clone(),
            BackendHealth {
                backend_id: backend_id.clone(),
                is_healthy: true,
                last_check: Instant::now(),
                consecutive_failures: 0,
                response_time_ms: 0,
            },
        );

        Ok(())
    }

    /// Remove a backend
    pub fn remove_backend(&self, backend_id: &str) -> FusekiResult<()> {
        let mut backends = self.backends.write();
        backends.retain(|b| b.id != backend_id);
        self.backend_stats.remove(backend_id);
        self.backend_health.remove(backend_id);
        Ok(())
    }

    /// Select a backend for the next request
    pub fn select_backend(
        &self,
        client_ip: Option<&str>,
        session_id: Option<&str>,
    ) -> FusekiResult<Backend> {
        // Check for sticky session
        if self.config.enable_sticky_sessions {
            if let Some(session_id) = session_id {
                if let Some(entry) = self.session_affinity.get(session_id) {
                    if let Some(backend) = self.get_backend_by_id(entry.value()) {
                        if backend.enabled && self.is_backend_healthy(&backend.id) {
                            return Ok(backend);
                        }
                    }
                }
            }
        }

        // Get healthy backends
        let healthy_backends = self.get_healthy_backends();
        if healthy_backends.is_empty() {
            return Err(FusekiError::service_unavailable(
                "No healthy backends available",
            ));
        }

        // Select based on strategy
        let selected = match self.config.strategy {
            LoadBalancingStrategy::RoundRobin => self.select_round_robin(&healthy_backends),
            LoadBalancingStrategy::WeightedRoundRobin => {
                self.select_weighted_round_robin(&healthy_backends)
            }
            LoadBalancingStrategy::LeastConnections => {
                self.select_least_connections(&healthy_backends)
            }
            LoadBalancingStrategy::LeastResponseTime => {
                self.select_least_response_time(&healthy_backends)
            }
            LoadBalancingStrategy::Random => self.select_random(&healthy_backends),
            LoadBalancingStrategy::WeightedRandom => self.select_weighted_random(&healthy_backends),
            LoadBalancingStrategy::IpHash => {
                self.select_ip_hash(&healthy_backends, client_ip.unwrap_or(""))
            }
            LoadBalancingStrategy::ConsistentHash => {
                self.select_consistent_hash(&healthy_backends, client_ip.unwrap_or(""))
            }
            LoadBalancingStrategy::PowerOfTwoChoices => self.select_power_of_two(&healthy_backends),
        }?;

        // Store session affinity if enabled
        if self.config.enable_sticky_sessions {
            if let Some(session_id) = session_id {
                self.session_affinity
                    .insert(session_id.to_string(), selected.id.clone());
            }
        }

        Ok(selected)
    }

    /// Record a request completion
    pub fn record_request(
        &self,
        backend_id: &str,
        response_time: Duration,
        is_error: bool,
    ) -> FusekiResult<()> {
        if let Some(stats) = self.backend_stats.get(backend_id) {
            stats.record_request(response_time.as_millis() as u64, is_error);

            // Update health status based on errors
            if is_error {
                if let Some(mut health) = self.backend_health.get_mut(backend_id) {
                    health.consecutive_failures += 1;
                    if health.consecutive_failures >= self.config.max_failures_before_unhealthy {
                        health.is_healthy = false;
                    }
                }
            } else if let Some(mut health) = self.backend_health.get_mut(backend_id) {
                health.consecutive_failures = 0;
                health.is_healthy = true;
                health.response_time_ms = response_time.as_millis() as u64;
            }
        }

        Ok(())
    }

    /// Acquire a connection to a backend
    pub fn acquire_connection(&self, backend_id: &str) -> FusekiResult<BackendConnection> {
        if let Some(stats) = self.backend_stats.get(backend_id) {
            stats.increment_connections();
            Ok(BackendConnection {
                backend_id: backend_id.to_string(),
                stats: Arc::clone(&stats),
            })
        } else {
            Err(FusekiError::not_found("Backend not found"))
        }
    }

    // Selection strategies

    fn select_round_robin(&self, backends: &[Backend]) -> FusekiResult<Backend> {
        let index = self.round_robin_index.fetch_add(1, Ordering::Relaxed) % backends.len();
        Ok(backends[index].clone())
    }

    fn select_weighted_round_robin(&self, backends: &[Backend]) -> FusekiResult<Backend> {
        let total_weight: u32 = backends.iter().map(|b| b.weight).sum();
        if total_weight == 0 {
            return self.select_round_robin(backends);
        }

        let mut index =
            self.round_robin_index.fetch_add(1, Ordering::Relaxed) % total_weight as usize;

        for backend in backends {
            if index < backend.weight as usize {
                return Ok(backend.clone());
            }
            index -= backend.weight as usize;
        }

        // Fallback
        Ok(backends[0].clone())
    }

    fn select_least_connections(&self, backends: &[Backend]) -> FusekiResult<Backend> {
        backends
            .iter()
            .min_by_key(|b| {
                self.backend_stats
                    .get(&b.id)
                    .map(|s| s.get_active_connections())
                    .unwrap_or(usize::MAX)
            })
            .cloned()
            .ok_or_else(|| FusekiError::service_unavailable("No backends available"))
    }

    fn select_least_response_time(&self, backends: &[Backend]) -> FusekiResult<Backend> {
        backends
            .iter()
            .min_by(|a, b| {
                let a_time = self
                    .backend_stats
                    .get(&a.id)
                    .map(|s| s.get_avg_response_time())
                    .unwrap_or(f64::MAX);
                let b_time = self
                    .backend_stats
                    .get(&b.id)
                    .map(|s| s.get_avg_response_time())
                    .unwrap_or(f64::MAX);
                a_time
                    .partial_cmp(&b_time)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .ok_or_else(|| FusekiError::service_unavailable("No backends available"))
    }

    fn select_random(&self, backends: &[Backend]) -> FusekiResult<Backend> {
        if backends.is_empty() {
            return Err(FusekiError::service_unavailable("No backends available"));
        }
        let index = self.random.lock().gen_range(0..backends.len());
        Ok(backends[index].clone())
    }

    fn select_weighted_random(&self, backends: &[Backend]) -> FusekiResult<Backend> {
        let total_weight: u32 = backends.iter().map(|b| b.weight).sum();
        if total_weight == 0 {
            return self.select_random(backends);
        }

        let mut choice = self.random.lock().gen_range(0..total_weight);
        for backend in backends {
            if choice < backend.weight {
                return Ok(backend.clone());
            }
            choice -= backend.weight;
        }

        // Fallback
        Ok(backends[0].clone())
    }

    fn select_ip_hash(&self, backends: &[Backend], client_ip: &str) -> FusekiResult<Backend> {
        if backends.is_empty() {
            return Err(FusekiError::service_unavailable("No backends available"));
        }

        // Simple hash of client IP
        let hash = client_ip
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let index = (hash % backends.len() as u64) as usize;
        Ok(backends[index].clone())
    }

    fn select_consistent_hash(&self, backends: &[Backend], key: &str) -> FusekiResult<Backend> {
        if backends.is_empty() {
            return Err(FusekiError::service_unavailable("No backends available"));
        }

        // Simple consistent hash (in production, use a proper consistent hash ring)
        let hash = self.hash_key(key);

        // Find the backend with the closest hash
        backends
            .iter()
            .min_by_key(|b| {
                let backend_hash = self.hash_key(&b.id);
                if hash >= backend_hash {
                    hash - backend_hash
                } else {
                    u64::MAX - (backend_hash - hash)
                }
            })
            .cloned()
            .ok_or_else(|| FusekiError::service_unavailable("No backends available"))
    }

    fn select_power_of_two(&self, backends: &[Backend]) -> FusekiResult<Backend> {
        if backends.is_empty() {
            return Err(FusekiError::service_unavailable("No backends available"));
        }

        if backends.len() == 1 {
            return Ok(backends[0].clone());
        }

        // Pick 2 random backends
        let mut rng = self.random.lock();
        let idx1 = rng.gen_range(0..backends.len());
        let idx2 = rng.gen_range(0..backends.len());
        drop(rng); // Release lock early

        let backend1 = &backends[idx1];
        let backend2 = &backends[idx2];

        // Choose the one with fewer connections
        let conn1 = self
            .backend_stats
            .get(&backend1.id)
            .map(|s| s.get_active_connections())
            .unwrap_or(usize::MAX);
        let conn2 = self
            .backend_stats
            .get(&backend2.id)
            .map(|s| s.get_active_connections())
            .unwrap_or(usize::MAX);

        if conn1 <= conn2 {
            Ok(backend1.clone())
        } else {
            Ok(backend2.clone())
        }
    }

    // Helper methods

    fn get_healthy_backends(&self) -> Vec<Backend> {
        self.backends
            .read()
            .iter()
            .filter(|b| b.enabled && self.is_backend_healthy(&b.id))
            .cloned()
            .collect()
    }

    fn is_backend_healthy(&self, backend_id: &str) -> bool {
        self.backend_health
            .get(backend_id)
            .map(|h| h.is_healthy)
            .unwrap_or(false)
    }

    fn get_backend_by_id(&self, backend_id: &str) -> Option<Backend> {
        self.backends
            .read()
            .iter()
            .find(|b| b.id == backend_id)
            .cloned()
    }

    fn hash_key(&self, key: &str) -> u64 {
        // Simple FNV-1a hash
        let mut hash = 14695981039346656037u64;
        for byte in key.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(1099511628211);
        }
        hash
    }

    /// Get statistics for all backends
    pub fn get_statistics(&self) -> HashMap<String, BackendStatistics> {
        let mut stats = HashMap::new();

        for entry in self.backend_stats.iter() {
            let backend_id = entry.key().clone();
            let backend_stats = entry.value();

            let health = self.backend_health.get(&backend_id);

            stats.insert(
                backend_id.clone(),
                BackendStatistics {
                    backend_id,
                    active_connections: backend_stats.get_active_connections(),
                    total_requests: backend_stats.total_requests.load(Ordering::Relaxed),
                    total_errors: backend_stats.total_errors.load(Ordering::Relaxed),
                    avg_response_time_ms: backend_stats.get_avg_response_time(),
                    is_healthy: health.as_ref().map(|h| h.is_healthy).unwrap_or(false),
                    consecutive_failures: health
                        .as_ref()
                        .map(|h| h.consecutive_failures)
                        .unwrap_or(0),
                },
            );
        }

        stats
    }
}

/// Backend connection guard (automatically decrements connection count on drop)
pub struct BackendConnection {
    backend_id: String,
    stats: Arc<BackendStats>,
}

impl Drop for BackendConnection {
    fn drop(&mut self) {
        self.stats.decrement_connections();
    }
}

/// Backend statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct BackendStatistics {
    pub backend_id: String,
    pub active_connections: usize,
    pub total_requests: u64,
    pub total_errors: u64,
    pub avg_response_time_ms: f64,
    pub is_healthy: bool,
    pub consecutive_failures: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_backend(id: &str, weight: u32) -> Backend {
        Backend {
            id: id.to_string(),
            url: format!("http://backend-{}", id),
            weight,
            max_connections: 100,
            health_check_url: None,
            enabled: true,
        }
    }

    #[test]
    fn test_add_remove_backend() {
        let lb = LoadBalancer::new(LoadBalancerConfig::default());

        let backend = create_test_backend("backend1", 1);
        lb.add_backend(backend).unwrap();

        assert_eq!(lb.backends.read().len(), 1);

        lb.remove_backend("backend1").unwrap();
        assert_eq!(lb.backends.read().len(), 0);
    }

    #[test]
    fn test_round_robin() {
        let config = LoadBalancerConfig {
            strategy: LoadBalancingStrategy::RoundRobin,
            ..Default::default()
        };
        let lb = LoadBalancer::new(config);

        lb.add_backend(create_test_backend("backend1", 1)).unwrap();
        lb.add_backend(create_test_backend("backend2", 1)).unwrap();
        lb.add_backend(create_test_backend("backend3", 1)).unwrap();

        let b1 = lb.select_backend(None, None).unwrap();
        let b2 = lb.select_backend(None, None).unwrap();
        let b3 = lb.select_backend(None, None).unwrap();
        let b4 = lb.select_backend(None, None).unwrap();

        assert_eq!(b1.id, "backend1");
        assert_eq!(b2.id, "backend2");
        assert_eq!(b3.id, "backend3");
        assert_eq!(b4.id, "backend1"); // Should wrap around
    }

    #[test]
    fn test_least_connections() {
        let config = LoadBalancerConfig {
            strategy: LoadBalancingStrategy::LeastConnections,
            ..Default::default()
        };
        let lb = LoadBalancer::new(config);

        lb.add_backend(create_test_backend("backend1", 1)).unwrap();
        lb.add_backend(create_test_backend("backend2", 1)).unwrap();

        // Acquire connection to backend1
        let _conn1 = lb.acquire_connection("backend1").unwrap();

        // Next request should go to backend2 (fewer connections)
        let backend = lb.select_backend(None, None).unwrap();
        assert_eq!(backend.id, "backend2");
    }

    #[test]
    fn test_ip_hash_consistency() {
        let config = LoadBalancerConfig {
            strategy: LoadBalancingStrategy::IpHash,
            ..Default::default()
        };
        let lb = LoadBalancer::new(config);

        lb.add_backend(create_test_backend("backend1", 1)).unwrap();
        lb.add_backend(create_test_backend("backend2", 1)).unwrap();

        // Same IP should always go to same backend
        let b1 = lb.select_backend(Some("192.168.1.1"), None).unwrap();
        let b2 = lb.select_backend(Some("192.168.1.1"), None).unwrap();
        assert_eq!(b1.id, b2.id);

        // Different IP might go to different backend
        let b3 = lb.select_backend(Some("192.168.1.2"), None).unwrap();
        // Not necessarily different, but we can at least check it returns something
        assert!(!b3.id.is_empty());
    }

    #[test]
    fn test_health_tracking() {
        let lb = LoadBalancer::new(LoadBalancerConfig::default());
        lb.add_backend(create_test_backend("backend1", 1)).unwrap();

        // Record successful requests
        lb.record_request("backend1", Duration::from_millis(50), false)
            .unwrap();

        let stats = lb.get_statistics();
        let backend_stats = stats.get("backend1").unwrap();
        assert_eq!(backend_stats.total_requests, 1);
        assert_eq!(backend_stats.total_errors, 0);
        assert!(backend_stats.is_healthy);
    }

    #[test]
    fn test_connection_counting() {
        let lb = LoadBalancer::new(LoadBalancerConfig::default());
        lb.add_backend(create_test_backend("backend1", 1)).unwrap();

        {
            let _conn1 = lb.acquire_connection("backend1").unwrap();
            let stats = lb.get_statistics();
            assert_eq!(stats.get("backend1").unwrap().active_connections, 1);

            let _conn2 = lb.acquire_connection("backend1").unwrap();
            let stats = lb.get_statistics();
            assert_eq!(stats.get("backend1").unwrap().active_connections, 2);
        }

        // Connections should be released
        let stats = lb.get_statistics();
        assert_eq!(stats.get("backend1").unwrap().active_connections, 0);
    }
}
